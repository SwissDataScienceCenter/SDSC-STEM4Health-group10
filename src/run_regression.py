"""Cross-validate the per-time regression model and write a leaderboard submission.

Usage (from the repo root):
    python src/run_regression.py [--seeds 0 1 2] [--simple [--blend 0.2]] [--no-fpca] [--name ridge_ensemble]

Model: per (target, sampling index) ridge regression on log targets (src/regression.py), using only the
t = 0 state, the volume and the Feed B / substrate feed profile; predicted vcd / tcd / byproduct trajectories
projected on their 2 main modes of variation (functional PCA); geometrically blended, per target, with the
VCD0-scaled mean-curve baseline and a nearest-batches baseline (R.BLEND, penalty R.ALPHA_SCALE); substrate
predictions corrected with the controller's extra-bolus rule (src/control_rules.py).
--simple: previous version (one blend weight with the scaled baseline, no ensemble, no rule).

Outputs in results/<name>/:
    cv_mape.csv            MAPE (%) per target, per CV seed (7 folds by batch) and averaged
    oof_predictions.csv    out-of-fold predictions (seed 0)
    coefficients_<v>.csv   standardised ridge coefficients per sampling index (fit on all 49 batches)
and submissions/<name>.csv: predictions for the 12 test batches in the template's row order.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # figures are saved to files, no display
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B
import control_rules as C
import ode_paramnet as O
import data as D
import evaluation as E
import regression as R

REPO = Path(__file__).resolve().parents[1]
# shared v3 parameters (fitted on all 49 training batches; using them inside CV is very mildly optimistic:
# on seed 0 the ODE scores 7.014 % with them vs 7.02 % with fold-wise refits)
V3_SHARED = REPO / "results" / "v3" / "params_full.csv"


def predict(raw, table, features, train_ids, ids, blend, n_eig=None, ensemble=False, rules=False, ode=False):
    """Fit on train_ids, predict ids (sorted by batch and time).

    ensemble=False: ridge blended with the scaled baseline (weight `blend`), same for every target.
    ensemble=True : per-target penalty (R.ALPHA_SCALE) and per-target geometric blend (R.BLEND) of
                    ridge / VCD0-scaled mean curve / nearest-batches baseline.
    rules=True    : substrate control-rule post-processing (src/control_rules.py).
    ode=True      : geometric blend with the hybrid ODE (weights R.ODE_WEIGHT, src/ode_paramnet.py)."""
    key = ["batch_id", "elapsed_time"]
    srt = lambda d: d.sort_values(key).reset_index(drop=True)
    make = {v: R.scaled_ridge(k) for v, k in R.ALPHA_SCALE.items()} if ensemble else R.ridge
    p = srt(R.PerTimeRegression(features, make, n_eigencurves=n_eig).fit(table, train_ids).predict(table, ids))
    b = srt(B.MeanTrajectory(scale_cells=True).fit(raw, train_ids).predict(raw, ids))
    assert (p[key].to_numpy() == b[key].to_numpy()).all()
    if ensemble:
        nb = srt(R.NearestBatches().fit(table, train_ids).predict(table, ids))
        for v in R.TARGETS:
            w = R.BLEND[v]
            p[v] = np.exp(w[0] * np.log(p[v]) + w[1] * np.log(b[v]) + w[2] * np.log(nb[v]))
    elif blend > 0:
        for v in R.TARGETS:
            p[v] = np.exp((1 - blend) * np.log(p[v]) + blend * np.log(b[v]))
    if ode:
        shared = pd.read_csv(V3_SHARED, index_col=0)["estimate"].to_dict()
        o = srt(O.predict_ode(raw, table, train_ids, ids, shared, R.ANALOG_FEATURES))
        for v in R.TARGETS:
            w = R.ODE_WEIGHT[v]
            if w > 0:
                p[v] = np.exp((1 - w) * np.log(p[v]) + w * np.log(o[v]))
    if rules:
        p = C.SubstrateControlRule().fit(raw, train_ids).apply(p)
    p["dcd"] = p["tcd"] - p["vcd"]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--blend", type=float, default=0.2, help="weight of the scaled baseline (0 = ridge only)")
    ap.add_argument("--no-fpca", action="store_true", help="no FPCA projection of the trajectories")
    ap.add_argument("--simple", action="store_true", help="single blend weight, no ensemble, no control rules")
    ap.add_argument("--no-ode", action="store_true", help="no hybrid ODE in the blend")
    ap.add_argument("--name", default="hybrid_ensemble")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    raw = D.load_raw()
    train_ids, test_ids = D.train_batches(raw), D.test_batches(raw)
    table = R.build_table(raw)
    features = R.FEATURES_BY_TARGET
    n_eig = None if a.no_fpca else R.N_EIGENCURVES
    opts = dict(ensemble=not a.simple, rules=not a.simple, ode=not (a.simple or a.no_ode))
    out = REPO / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    for v, cols in features.items():
        print(f"[{a.name}] {v}: {len(cols)} features: {cols}")

    # Cross-validation: 7 folds by batch, repeated over several random fold assignments
    meas, scores = D.measurements(raw), {}
    for s in a.seeds:
        folds = E.make_folds(train_ids, 7, s)
        oof = pd.concat([predict(raw, table, features, [b for b in train_ids if b not in f], f, a.blend, n_eig, **opts)
                         for f in folds], ignore_index=True)
        truth = E.attach_truth(oof.drop(columns="j"), meas)
        scores[f"seed{s}"] = E.mape_summary(truth)
        if s == a.seeds[0]:
            # Rule from CLAUDE.md: RMSE + relative errors per variable and predicted-vs-measured plots
            oof.to_csv(out / "oof_predictions.csv", index=False)
            E.metrics(truth).to_csv(out / "cv_metrics.csv")
            print(f"[{a.name}] seed {s}: RMSE / relative errors per variable\n{E.metrics(truth).round(3).to_string()}")
            worst = truth.groupby("batch_id").apply(
                lambda g: np.mean([np.mean(np.abs(g[f"{v}_pred"] - g[v]) / g[v]) for v in R.TARGETS])).sort_values()
            show = [worst.index[0], worst.index[len(worst) // 2], worst.index[-2], worst.index[-1]]  # best, median, 2 worst
            E.plot_pred_vs_meas(truth, show).savefig(out / "pred_vs_meas.png", bbox_inches="tight", dpi=90)
            E.parity_plot(truth, title=f"{a.name}: out-of-fold predictions").savefig(out / "parity.png", bbox_inches="tight", dpi=90)
    scores = pd.DataFrame(scores)
    scores["mean"] = scores.mean(axis=1)
    scores.to_csv(out / "cv_mape.csv")
    print(f"\n[{a.name}] cross-validation MAPE (%), 7 folds by batch:\n{scores.round(2).to_string()}")

    # Final model: fit on all 49 training batches, predict the 12 test batches
    model = R.PerTimeRegression(features, n_eigencurves=n_eig).fit(table, train_ids)
    for v in R.TARGETS:
        model.coefficients(v).to_csv(out / f"coefficients_{v}.csv")
    pred = predict(raw, table, features, train_ids, test_ids, a.blend, n_eig, **opts)
    tpl = D.load_template()
    k = lambda d: d.assign(_t=d["elapsed_time"].round(6))
    sub = (k(tpl)[["batch_id", "elapsed_time", "_t"]]
           .merge(k(pred)[["batch_id", "_t", *R.TARGETS]], on=["batch_id", "_t"], how="left").drop(columns="_t"))
    assert len(sub) == len(tpl) and sub[R.TARGETS].notna().all().all() and (sub[R.TARGETS] > 0).all().all()
    assert (sub[["batch_id", "elapsed_time"]].to_numpy() == tpl[["batch_id", "elapsed_time"]].to_numpy()).all()
    (REPO / "submissions").mkdir(exist_ok=True)
    sub.to_csv(REPO / "submissions" / f"{a.name}.csv", index=False)
    print(f"\n[{a.name}] submission written: submissions/{a.name}.csv ({len(sub)} rows)")


if __name__ == "__main__":
    main()

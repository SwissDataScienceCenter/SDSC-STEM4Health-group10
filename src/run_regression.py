"""Cross-validate the per-time regression model and write a leaderboard submission.

Usage (from the repo root):
    python src/run_regression.py [--seeds 0 1 2] [--blend 0.2] [--name ridge_blend]

Model: per (target, sampling index) ridge regression on log targets (src/regression.py), using only the
t = 0 state, the volume and the Feed B / substrate feed profile; geometrically blended with the
VCD0-scaled mean-curve baseline:  pred = ridge^(1-w) * baseline^w.

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
import data as D
import evaluation as E
import regression as R

REPO = Path(__file__).resolve().parents[1]


def predict(raw, table, features, train_ids, ids, blend):
    """Fit on train_ids, predict ids (sorted by batch and time), blended with the scaled baseline."""
    key = ["batch_id", "elapsed_time"]
    p = R.PerTimeRegression(features).fit(table, train_ids).predict(table, ids).sort_values(key).reset_index(drop=True)
    if blend > 0:
        b = B.MeanTrajectory(scale_cells=True).fit(raw, train_ids).predict(raw, ids).sort_values(key).reset_index(drop=True)
        assert (p[key].to_numpy() == b[key].to_numpy()).all()
        for v in R.TARGETS:
            p[v] = np.exp((1 - blend) * np.log(p[v]) + blend * np.log(b[v]))
    p["dcd"] = p["tcd"] - p["vcd"]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--blend", type=float, default=0.2, help="weight of the scaled baseline (0 = ridge only)")
    ap.add_argument("--name", default="ridge_blend")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    raw = D.load_raw()
    train_ids, test_ids = D.train_batches(raw), D.test_batches(raw)
    table = R.build_table(raw)
    features = R.FEATURES_BY_TARGET
    out = REPO / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    for v, cols in features.items():
        print(f"[{a.name}] {v}: {len(cols)} features: {cols}")

    # Cross-validation: 7 folds by batch, repeated over several random fold assignments
    meas, scores = D.measurements(raw), {}
    for s in a.seeds:
        folds = E.make_folds(train_ids, 7, s)
        oof = pd.concat([predict(raw, table, features, [b for b in train_ids if b not in f], f, a.blend)
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
    model = R.PerTimeRegression(features).fit(table, train_ids)
    for v in R.TARGETS:
        model.coefficients(v).to_csv(out / f"coefficients_{v}.csv")
    pred = predict(raw, table, features, train_ids, test_ids, a.blend)
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

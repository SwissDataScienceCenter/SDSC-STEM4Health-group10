"""Non-linear 'pooled-time' models for the cell densities, blended with the team submission.

Pooled-time model: ONE model per target, fitted on all (batch, sampling index j) rows of the training batches
(~550 rows instead of 42 per time point for the per-time ridge), with the sampling time as an input. Target =
log(y) - mu_j, the deviation from the mean log trajectory of the training batches at index j: the learners only
model how a batch departs from the typical curve (trees cannot extrapolate a time trend).

Families benchmarked (src/experiments/families_cv.py, same 7 folds by batch as the team models, 6 seeds):
LightGBM, random forest, extra-trees, kernel ridge (RBF), SVR (RBF), small MLP, each on two feature sets.
  * substrate / byproduct: all much worse than the per-time models (9.5-11 % vs 6.9 / 8.7 %) and no gain in a
    blend -> not used (feed boluses create jumps at specific times that per-time models capture better).
  * vcd / tcd: SVR alone is as good as the team blend (6.60 / 6.54 vs 6.61 / 6.55 %) with different errors.
    Blend (geometric): 0.25 team + 0.55 kernel group (SVR x2) + 0.20 tree group (RF, LightGBM, extra-trees),
    weights chosen on seeds 0-2: gain 0.108 (vcd) and 0.114 (tcd) MAPE points on the FRESH seeds 3-5.

Usage (from the repo root, after src/blend_team.py):
    python src/model_families.py        # -> submissions/blend_nonlinear.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import regression as R

REPO = Path(__file__).resolve().parents[1]

# Feature sets: "sel" = per-target ridge selection (+ time), "wide" = every feature except Feed C
# (Feed C features failed twice on the leaderboard).
FEED_C = ("fc_t_first", "nC_early", "nC_late_log")


def feature_sets(table: pd.DataFrame) -> dict:
    wide = [c for c in R.feature_columns(table) if "feed_c" not in c and c not in FEED_C]
    return {"sel": {v: sorted(set(R.FEATURES_BY_TARGET[v]) | {"t"}) for v in R.TARGETS},
            "wide": {v: wide for v in R.TARGETS}}


def _grid(est, params):
    # hyper-parameters chosen by internal cross-validation grouped by batch (no leakage between batches)
    return lambda: GridSearchCV(est, params, cv=GroupKFold(5), scoring="neg_mean_squared_error")


FAMILIES = {
    "lgbm": lambda: LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  n_jobs=1, verbose=-1, random_state=0),
    "extratrees": lambda: ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, n_jobs=1,
                                              random_state=0),
    "rf": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, n_jobs=1,
                                        random_state=0),
    "krr": _grid(make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                 {"kernelridge__alpha": [0.01, 0.1, 1.0], "kernelridge__gamma": [0.003, 0.01, 0.03, 0.1]}),
    "svr": _grid(make_pipeline(StandardScaler(), SVR(kernel="rbf")),
                 {"svr__C": [0.3, 1, 3], "svr__gamma": [0.003, 0.01, 0.03], "svr__epsilon": [0.01, 0.03]}),
    "mlp": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.1,
                                                                learning_rate_init=3e-3, max_iter=3000,
                                                                early_stopping=True, random_state=0)),
}

# Final blend for vcd / tcd (geometric weights): team submission, kernel group, tree group.
KERNEL_GROUP = ["svr_sel", "svr_wide"]
TREE_GROUP = ["rf_sel", "lgbm_sel", "extratrees_sel"]
W_FINAL = {"team": 0.25, "kernel": 0.55, "trees": 0.20}
CELL_TARGETS = ["vcd", "tcd"]


def pooled_predict(table: pd.DataFrame, train_ids, ids, family: str, fset: str, targets=R.TARGETS) -> pd.DataFrame:
    """Fit one pooled-time model per target on train_ids, predict ids. Returns batch_id, j, elapsed_time, targets."""
    cols = feature_sets(table)[fset]
    tr, te = table[table.batch_id.isin(train_ids)], table[table.batch_id.isin(ids)]
    out = te[["batch_id", "j", "elapsed_time"]].copy()
    for v in targets:
        mu = np.log(tr[v]).groupby(tr.j).mean()                 # mean log trajectory (training batches)
        y = np.log(tr[v]) - tr.j.map(mu)
        m = FAMILIES[family]()
        if family in ("krr", "svr"):
            m.fit(tr[cols[v]], y, groups=tr.batch_id)
        else:
            m.fit(tr[cols[v]], y)
        out[v] = np.exp(te.j.map(mu) + m.predict(te[cols[v]]))
    return out.reset_index(drop=True)


def geo_mean(frames: list[pd.DataFrame], v: str) -> np.ndarray:
    return np.exp(np.mean([np.log(f[v].to_numpy()) for f in frames], axis=0))


def main():
    warnings.filterwarnings("ignore")
    raw = D.load_raw()
    table = R.build_table(raw)
    train_ids, test_ids = D.train_batches(raw), D.test_batches(raw)
    tpl = D.load_template()
    team = pd.read_csv(REPO / "submissions" / "blend_team.csv")
    assert (team.batch_id.to_numpy() == tpl.batch_id.to_numpy()).all() and np.allclose(team.elapsed_time, tpl.elapsed_time)

    preds = {}
    for name in KERNEL_GROUP + TREE_GROUP:
        family, fset = name.rsplit("_", 1)
        p = pooled_predict(table, train_ids, test_ids, family, fset, CELL_TARGETS)
        k = lambda d: d.assign(_t=d["elapsed_time"].round(6))
        p = k(tpl)[["batch_id", "_t"]].merge(k(p)[["batch_id", "_t", *CELL_TARGETS]], on=["batch_id", "_t"], how="left")
        assert p[CELL_TARGETS].notna().all().all()
        preds[name] = p
        print(f"fitted {name}")

    sub = team.copy()
    for v in CELL_TARGETS:
        kern = geo_mean([preds[n] for n in KERNEL_GROUP], v)
        trees = geo_mean([preds[n] for n in TREE_GROUP], v)
        sub[v] = np.exp(W_FINAL["team"] * np.log(team[v]) + W_FINAL["kernel"] * np.log(kern)
                        + W_FINAL["trees"] * np.log(trees))
    sub["tcd"] = np.maximum(sub.tcd, sub.vcd)
    assert len(sub) == len(tpl) and sub[R.TARGETS].notna().all().all() and (sub[R.TARGETS] > 0).all().all()
    out = REPO / "submissions" / "blend_nonlinear.csv"
    sub.to_csv(out, index=False)
    ch = (100 * (sub[R.TARGETS] - team[R.TARGETS]).abs() / team[R.TARGETS]).median().round(2).to_dict()
    print(f"submission written: {out.relative_to(REPO)} | median change vs blend_team (%): {ch}")


if __name__ == "__main__":
    main()

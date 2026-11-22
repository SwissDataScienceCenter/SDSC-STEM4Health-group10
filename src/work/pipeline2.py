"""Stacking pipeline v2: log-target base models, more learners, per-day-bucket
meta-learner, ODE-sim features, trajectory-kNN base. Honest GroupKFold CV.

Usage:
  python pipeline2.py --cv [--folds 6] [--no-ode]
  python pipeline2.py --submit out.csv [--folds 6] [--no-ode]
"""
import sys, os, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import load_batches, SimSet
from pipeline import (build_rows, add_knn_feats, knn_traj_feature, batch_schedule_feats,
                      BASE_FEATS, ODE_FEATS, KNN_FEATS, TARGETS)

from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
import lightgbm as lgb
import xgboost as xgb


def make_models(seed=0):
    return {
        "hgb": HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.045, max_leaf_nodes=31,
                                             min_samples_leaf=5, l2_regularization=1.0,
                                             random_state=seed),
        "hgb2": HistGradientBoostingRegressor(max_iter=900, learning_rate=0.03, max_leaf_nodes=63,
                                              min_samples_leaf=4, l2_regularization=0.5,
                                              random_state=seed + 1),
        "et": ExtraTreesRegressor(n_estimators=700, min_samples_leaf=3, max_features=0.7,
                                  random_state=seed, n_jobs=8),
        "rf": RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.7,
                                    random_state=seed, n_jobs=8),
        "lgbm": lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.04, num_leaves=31,
                                  min_child_samples=5, reg_lambda=1.0, random_state=seed,
                                  n_jobs=8, verbose=-1),
        "xgb": xgb.XGBRegressor(n_estimators=1000, learning_rate=0.04, max_depth=6,
                                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                                reg_lambda=1.0, random_state=seed, n_jobs=8),
        "knn": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=8, weights="distance")),
    }


DAY_BUCKETS = [(0, 3), (4, 7), (8, 10), (11, 13)]


class LogWrap:
    """Fit base model on log(y), predict exp - MAPE friendly."""

    def __init__(self, mdl):
        self.mdl = mdl

    def fit(self, X, y):
        self.mdl.fit(X, np.log(np.maximum(y, 1e-6)))
        return self

    def predict(self, X):
        return np.exp(self.mdl.predict(X))


def traj_knn_pred(B, ref_ids, tgt_ids, k=5):
    """Returns dict tgt -> (n_rows_max, 4) predictions [vcd,tcd,S,P] by obs index."""
    traj = knn_traj_feature(B, ref_ids, tgt_ids, k=k)
    return traj


def run_cv(R, B, use_ode, folds, knn_k=5, verbose=True):
    train = R[~R.is_tp].copy().reset_index(drop=True)
    feats = list(BASE_FEATS) + (ODE_FEATS if use_ode else []) + KNN_FEATS
    model_names = list(make_models().keys()) + ["trajknn"]
    gkf = GroupKFold(n_splits=folds)
    base_oof = {t: {m: np.zeros(len(train)) for m in model_names} for t in TARGETS}
    for tr_idx, te_idx in gkf.split(train, groups=train.batch_id.values):
        tr = train.iloc[tr_idx].copy()
        te = train.iloc[te_idx].copy()
        ref_ids = sorted(tr.batch_id.unique(), key=lambda s: int(s[5:]))
        for b in ref_ids:
            others = [x for x in ref_ids if x != b]
            add_knn_feats(tr, B, tr.batch_id == b, others, k=knn_k)
        add_knn_feats(te, B, te.index == te.index, ref_ids, k=knn_k)
        te_ids = sorted(te.batch_id.unique(), key=lambda s: int(s[5:]))
        ttraj = traj_knn_pred(B, ref_ids, te_ids, k=knn_k)
        for t, ci in zip(TARGETS, range(4)):
            for name in model_names[:-1]:
                mdl = LogWrap(make_models()[name])
                mdl.fit(tr[feats], tr[t])
                base_oof[t][name][te_idx] = mdl.predict(te[feats])
            # trajknn base
            vals = []
            for _, r in te.iterrows():
                trj = ttraj[r.batch_id]
                vals.append(trj[min(int(r.k), len(trj) - 1), ci])
            base_oof[t]["trajknn"][te_idx] = np.array(vals)
    results = {}
    for t in TARGETS:
        y = train[t].values
        X = np.column_stack([base_oof[t][m] for m in model_names])
        # global NNLS
        w, _ = nnls(np.column_stack([X, np.ones(len(X))]), y)
        oof = X @ w[:-1] + w[-1]
        mape = (np.abs(oof - y) / y).mean() * 100
        # per-bucket NNLS
        oof_b = oof.copy()
        ks = train["k"].values
        for lo, hi in DAY_BUCKETS:
            m = (ks >= lo) & (ks <= hi)
            if m.sum() < 20:
                continue
            wb, _ = nnls(np.column_stack([X[m], np.ones(m.sum())]), y[m])
            oof_b[m] = X[m] @ wb[:-1] + wb[-1]
        mape_b = (np.abs(oof_b - y) / y).mean() * 100
        base_mapes = {m: (np.abs(base_oof[t][m] - y) / y).mean() * 100 for m in model_names}
        results[t] = dict(stack=mape, stack_bucket=mape_b, base=base_mapes)
        if verbose:
            print(f"{t:10s} stack {mape:6.2f} bucket {mape_b:6.2f} | " +
                  " ".join(f"{m}:{v:.2f}" for m, v in base_mapes.items()), flush=True)
    mean = np.mean([results[t]["stack_bucket"] for t in TARGETS])
    if verbose:
        print(f"CV mean MAPE (bucket stack): {mean:.3f}")
    return results, mean, base_oof, model_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--submit", default=None)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--no-ode", action="store_true")
    ap.add_argument("--ode-params", default=os.path.join(os.path.dirname(__file__), "params_ms.json"))
    args = ap.parse_args()

    B, tpl, df = load_batches()
    train_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    ode_params = None
    if not args.no_ode and os.path.exists(args.ode_params):
        ode_params = json.load(open(args.ode_params))["logp"]
    R = build_rows(B, tpl, ode_params=ode_params, knn_traj=None)
    for c in KNN_FEATS:
        R[c] = 0.0
    print(f"rows {len(R)} train {int((~R.is_tp).sum())} hidden {int(R.is_tp.sum())}", flush=True)

    if args.cv:
        results, mean, base_oof, model_names = run_cv(R, B, use_ode=not args.no_ode, folds=args.folds)
        print("mean", round(mean, 3))
        json.dump({t: {"stack": results[t]["stack"], "bucket": results[t]["stack_bucket"],
                       "base": results[t]["base"]} for t in TARGETS},
                  open(os.path.join(os.path.dirname(__file__), "cv_results2.json"), "w"), indent=1)

    if args.submit:
        train = R[~R.is_tp].copy().reset_index(drop=True)
        hid = R[R.is_tp].copy().reset_index(drop=True)
        feats = list(BASE_FEATS) + (ODE_FEATS if not args.no_ode else []) + KNN_FEATS
        model_names = list(make_models().keys()) + ["trajknn"]
        for b in train_ids:
            others = [x for x in train_ids if x != b]
            add_knn_feats(train, B, train.batch_id == b, others, k=5)
        add_knn_feats(hid, B, hid.index == hid.index, train_ids, k=5)
        gkf = GroupKFold(n_splits=args.folds)
        base_oof = {t: {m: np.zeros(len(train)) for m in model_names} for t in TARGETS}
        for tr_idx, te_idx in gkf.split(train, groups=train.batch_id.values):
            tr = train.iloc[tr_idx].copy()
            te = train.iloc[te_idx].copy()
            ref_ids = sorted(tr.batch_id.unique(), key=lambda s: int(s[5:]))
            for b in ref_ids:
                others = [x for x in ref_ids if x != b]
                add_knn_feats(tr, B, tr.batch_id == b, others, k=5)
            add_knn_feats(te, B, te.index == te.index, ref_ids, k=5)
            te_ids = sorted(te.batch_id.unique(), key=lambda s: int(s[5:]))
            ttraj = traj_knn_pred(B, ref_ids, te_ids, k=5)
            for t, ci in zip(TARGETS, range(4)):
                for name in model_names[:-1]:
                    mdl = LogWrap(make_models()[name])
                    mdl.fit(tr[feats], tr[t])
                    base_oof[t][name][te_idx] = mdl.predict(te[feats])
                vals = [ttraj[r.batch_id][min(int(r.k), len(ttraj[r.batch_id]) - 1), ci]
                        for _, r in te.iterrows()]
                base_oof[t]["trajknn"][te_idx] = np.array(vals)
        hid_ids = sorted(hid.batch_id.unique(), key=lambda s: int(s[5:]))
        htraj = traj_knn_pred(B, train_ids, hid_ids, k=5)
        out = hid[["batch_id", "t"]].copy()
        ks_h = hid["k"].values
        for t, ci in zip(TARGETS, range(4)):
            y = train[t].values
            X = np.column_stack([base_oof[t][m] for m in model_names])
            Xh = []
            for name in model_names[:-1]:
                mdl = LogWrap(make_models()[name])
                mdl.fit(train[feats], train[t])
                Xh.append(mdl.predict(hid[feats]))
            Xh.append(np.array([htraj[r.batch_id][min(int(r.k), len(htraj[r.batch_id]) - 1), ci]
                                for _, r in hid.iterrows()]))
            Xh = np.column_stack(Xh)
            # per-bucket weights
            pred = np.zeros(len(hid))
            ks = train["k"].values
            for lo, hi in DAY_BUCKETS:
                m = (ks >= lo) & (ks <= hi)
                mh = (ks_h >= lo) & (ks_h <= hi)
                if m.sum() < 20 or not mh.any():
                    w, _ = nnls(np.column_stack([X, np.ones(len(X))]), y)
                    pred[mh] = Xh[mh] @ w[:-1] + w[-1]
                    continue
                wb, _ = nnls(np.column_stack([X[m], np.ones(m.sum())]), y[m])
                pred[mh] = Xh[mh] @ wb[:-1] + wb[-1]
            out[t] = np.maximum(pred, 0.0)
        out = out.rename(columns={"t": "elapsed_time"})[["batch_id", "elapsed_time"] + TARGETS]
        out.to_csv(args.submit, index=False)
        print("wrote", args.submit)


if __name__ == "__main__":
    main()

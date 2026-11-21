"""Pipeline v4: GBM/LGBM/XGB row models + team-style kNN-trajectory base,
NNLS stacking per target (global + per-day-bucket). GroupKFold CV honest.

Usage:
  python pipeline4.py --cv [--folds 6] [--no-ode]
  python pipeline4.py --submit out.csv
"""
import sys, os, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import load_batches
from pipeline import build_rows, add_knn_feats, BASE_FEATS, ODE_FEATS, KNN_FEATS, TARGETS, batch_schedule_feats
from pipeline2 import make_models, LogWrap, DAY_BUCKETS

from sklearn.model_selection import GroupKFold
from scipy.optimize import nnls

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]


def rich_feats(B, b):
    bd = B[b]
    ft, fa = bd["f_t"], bd["f_a"]
    y0 = bd["y0"]
    days = np.arange(0.5, 13.5, 1.0)
    cumfb = np.array([fa[ft <= d, 1].sum() for d in days])
    cumfs = np.array([fa[ft <= d, 0].sum() for d in days])
    hist = np.histogram(ft[fa[:, 1] > 0], bins=np.arange(0, 14, 1),
                        weights=fa[fa[:, 1] > 0, 1])[0]
    return np.r_[[y0[0], y0[1], y0[2], y0[3], bd["v_v"][0]], cumfb, cumfs, hist]


def knn_traj_base(B, ref_ids, tgt_ids, k=5, power=2.0):
    F = np.stack([rich_feats(B, b) for b in ref_ids])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fn = (F - mu) / sd
    Obs = {b: np.column_stack([B[b]["y_obs"][:, 0],
                               B[b]["y_obs"][:, 0] + B[b]["y_obs"][:, 1],
                               B[b]["y_obs"][:, 2], B[b]["y_obs"][:, 3]]) for b in ref_ids}
    out = {}
    for b in tgt_ids:
        f = (rich_feats(B, b) - mu) / sd
        d = ((Fn - f) ** 2).sum(1)
        order = np.argsort(d)[:k]
        w = 1.0 / (d[order] ** power + 1e-12)
        w /= w.sum()
        out[b] = sum(w[i] * Obs[ref_ids[j]] for i, j in enumerate(order))
    return out


def base_models(seed=0):
    ms = make_models(seed)
    return {k: v for k, v in ms.items() if k in ("hgb", "hgb2", "et", "lgbm", "xgb")}


def run(R, B, use_ode, folds, knn_k=5, verbose=True, submit_path=None):
    train = R.loc[~R.is_tp].copy()
    hid = R.loc[R.is_tp].copy()
    all_train_ids = sorted(train.batch_id.unique(), key=lambda s: int(s[5:]))
    hid_ids = sorted(hid.batch_id.unique(), key=lambda s: int(s[5:]))
    feats = list(BASE_FEATS) + (ODE_FEATS if use_ode else []) + KNN_FEATS
    # LOO knn features for train, full knn for hidden
    for b in all_train_ids:
        add_knn_feats(train, B, train.batch_id == b, [x for x in all_train_ids if x != b], k=knn_k)
    add_knn_feats(hid, B, hid.batch_id == hid.batch_id, all_train_ids, k=knn_k)

    model_names = list(base_models(0).keys())
    gkf = GroupKFold(n_splits=folds)
    base_oof = {t: {m: np.zeros(len(train)) for m in model_names + ["knntraj"]} for t in TARGETS}
    base_hid = {t: {m: np.zeros(len(hid)) for m in model_names + ["knntraj"]} for t in TARGETS}
    for tr_pos, te_pos in gkf.split(train, groups=train.batch_id.values):
        tr = train.iloc[tr_pos].copy()
        te = train.iloc[te_pos].copy()
        ref_ids = sorted(tr.batch_id.unique(), key=lambda s: int(s[5:]))
        for b in ref_ids:
            add_knn_feats(tr, B, tr.batch_id == b, [x for x in ref_ids if x != b], k=knn_k)
        add_knn_feats(te, B, te.batch_id == te.batch_id, ref_ids, k=knn_k)
        te_ids = sorted(te.batch_id.unique(), key=lambda s: int(s[5:]))
        ttraj = knn_traj_base(B, ref_ids, te_ids, k=knn_k)
        for t, ci in zip(TARGETS, range(4)):
            for name in model_names:
                mdl = LogWrap(base_models(0)[name])
                mdl.fit(tr[feats], tr[t])
                base_oof[t][name][te_pos] = mdl.predict(te[feats])
            vals = np.array([ttraj[r.batch_id][min(int(r.k), 13), ci] for _, r in te.iterrows()])
            base_oof[t]["knntraj"][te_pos] = vals
    # full fits for hidden
    htraj = knn_traj_base(B, all_train_ids, hid_ids, k=knn_k)
    for t, ci in zip(TARGETS, range(4)):
        for name in model_names:
            mdl = LogWrap(base_models(0)[name])
            mdl.fit(train[feats], train[t])
            base_hid[t][name] = mdl.predict(hid[feats])
        base_hid[t]["knntraj"] = np.array([htraj[r.batch_id][min(int(r.k), 13), ci]
                                            for _, r in hid.iterrows()])
    # stacking
    results = {}
    stack_oof = {}
    for t in TARGETS:
        names = model_names + ["knntraj"]
        X = np.column_stack([base_oof[t][m] for m in names])
        Xh = np.column_stack([base_hid[t][m] for m in names])
        y = train[t].values
        ks = train["k"].values
        # log-space (geometric) NNLS stacking, per day-bucket
        Xl = np.log(np.maximum(X, 1e-6))
        Xhl = np.log(np.maximum(Xh, 1e-6))
        yl = np.log(y)
        w_all, _ = nnls(np.column_stack([Xl, np.ones(len(Xl))]), yl)
        oof = np.exp(Xl @ w_all[:-1] + w_all[-1])
        pred_h = np.exp(Xhl @ w_all[:-1] + w_all[-1])
        oof_b = oof.copy(); pred_b = pred_h.copy()
        ks_h = hid["k"].values
        for lo, hi in DAY_BUCKETS:
            m = (ks >= lo) & (ks <= hi)
            mh = (ks_h >= lo) & (ks_h <= hi)
            if m.sum() < 15:
                continue
            wb, _ = nnls(np.column_stack([Xl[m], np.ones(m.sum())]), yl[m])
            oof_b[m] = np.exp(Xl[m] @ wb[:-1] + wb[-1])
            if mh.any():
                pred_b[mh] = np.exp(Xhl[mh] @ wb[:-1] + wb[-1])
        mape = (np.abs(oof - y) / y).mean() * 100
        mape_b = (np.abs(oof_b - y) / y).mean() * 100
        bm = {m: (np.abs(base_oof[t][m] - y) / y).mean() * 100 for m in names}
        results[t] = dict(stack=mape, bucket=mape_b, base=bm)
        stack_oof[t] = (oof_b, pred_b, names)
        if verbose:
            print(f"{t:10s} stack {mape:6.2f} bucket {mape_b:6.2f} | " +
                  " ".join(f"{m}:{v:.2f}" for m, v in bm.items()), flush=True)
    mean = np.mean([results[t]["bucket"] for t in TARGETS])
    if verbose:
        print(f"CV mean (bucket): {mean:.3f}")
    if submit_path:
        out = hid[["batch_id", "t"]].copy()
        for t in TARGETS:
            out[t] = np.maximum(stack_oof[t][1], 0.0)
        out = out.rename(columns={"t": "elapsed_time"})[["batch_id", "elapsed_time"] + TARGETS]
        out.to_csv(submit_path, index=False)
        print("wrote", submit_path)
    return results, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--submit", default=None)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--no-ode", action="store_true")
    ap.add_argument("--ode-params", default=os.path.join(os.path.dirname(__file__), "params_ms.json"))
    args = ap.parse_args()

    B, tpl, df = load_batches()
    ode_params = None
    if not args.no_ode and os.path.exists(args.ode_params):
        ode_params = json.load(open(args.ode_params))["logp"]
    R = build_rows(B, tpl, ode_params=ode_params, knn_traj=None)
    for c in KNN_FEATS:
        R[c] = 0.0
    R = R.sort_values(["batch_id", "t"]).reset_index(drop=True)
    print(f"rows {len(R)}", flush=True)
    run(R, B, use_ode=not args.no_ode, folds=args.folds,
        submit_path=args.submit if args.submit else None)


if __name__ == "__main__":
    main()

"""Pipeline v3 (clean): physics-cascade features for substrate/byproduct.

S(t)V(t) = S0V0 + sum g_j cumF_j(t) - a*Jmu(t) - b*J(t)
P(t)V(t) = P0V0 + alpha*Jmu(t) + beta*J(t) - q*K(t)
J = int Xv V dt, Jmu = int mu Xv V dt, K = int Xv P/(KPc+P) V dt,
with Xv from the vcd ensemble (OOF for train rows, full-fit for hidden rows)
and P in K from a first-pass P ensemble (same OOF scheme).

Usage:
  python pipeline3.py --cv [--folds 6] [--no-ode]
  python pipeline3.py --submit out.csv
"""
import sys, os, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import load_batches
from pipeline import build_rows, add_knn_feats, BASE_FEATS, ODE_FEATS, KNN_FEATS, TARGETS
from pipeline2 import make_models, LogWrap, DAY_BUCKETS

from sklearn.model_selection import GroupKFold
from scipy.optimize import nnls

GROWTH = dict(mu_max=0.9, Ks=0.4, Kx=33.0)


def oof_predict(R, B, target, feats, folds, use_ode, knn_k=5, seed=0):
    """Returns (oof for train rows, preds for hidden rows) for target."""
    train = R.loc[~R.is_tp].copy()
    hid = R.loc[R.is_tp].copy()
    all_train_ids = sorted(train.batch_id.unique(), key=lambda s: int(s[5:]))
    for b in all_train_ids:
        others = [x for x in all_train_ids if x != b]
        add_knn_feats(train, B, train.batch_id == b, others, k=knn_k)
    add_knn_feats(hid, B, hid.batch_id == hid.batch_id, all_train_ids, k=knn_k)
    model_names = list(make_models(seed).keys())
    gkf = GroupKFold(n_splits=folds)
    base_oof = {m: np.zeros(len(train)) for m in model_names}
    groups = train.batch_id.values
    for tr_pos, te_pos in gkf.split(train, groups=groups):
        tr = train.iloc[tr_pos]
        te = train.iloc[te_pos]
        ref_ids = sorted(tr.batch_id.unique(), key=lambda s: int(s[5:]))
        for b in ref_ids:
            add_knn_feats(tr, B, tr.batch_id == b, [x for x in ref_ids if x != b], k=knn_k)
        add_knn_feats(te, B, te.batch_id == te.batch_id, ref_ids, k=knn_k)
        for name in model_names:
            mdl = LogWrap(make_models(seed)[name])
            mdl.fit(tr[feats], tr[target])
            base_oof[name][te_pos] = mdl.predict(te[feats])
    X = np.column_stack([base_oof[m] for m in model_names])
    y = train[target].values
    w, _ = nnls(np.column_stack([X, np.ones(len(X))]), y)
    oof = X @ w[:-1] + w[-1]
    Xh = []
    for name in model_names:
        mdl = LogWrap(make_models(seed)[name])
        mdl.fit(train[feats], train[target])
        Xh.append(mdl.predict(hid[feats]))
    Xh = np.column_stack(Xh)
    hid_pred = Xh @ w[:-1] + w[-1]
    out = pd.Series(np.nan, index=R.index)
    out.loc[train.index.values] = oof
    out.loc[hid.index.values] = hid_pred
    return out, w, model_names


def cascade_feats(R, B, vcd_map, P_map):
    """Add J, Jmu, cumF_fs, cumF_fb, cumF_fc, cumF_fd, K columns (positional per batch)."""
    R = R.copy()
    for c in ["J", "Jmu", "cumF_fs", "cumF_fb", "cumF_fc", "cumF_fd", "K"]:
        R[c] = 0.0
    for b, g in R.groupby("batch_id", sort=False):
        g = g.sort_values("t")
        idx = g.index.values
        t = g["t"].values
        V = g["V"].values
        Xv = np.asarray(vcd_map[b])
        Pv = np.asarray(P_map[b])
        bd = B[b]
        ft, fa = bd["f_t"], bd["f_a"]
        cumF = np.array([fa[ft <= tt + 1e-9].sum(0) if (ft <= tt + 1e-9).any() else np.zeros(4)
                         for tt in t])
        S_ode = g["ode_S"].values if "ode_S" in g.columns else np.full(len(t), bd["y0"][2])
        dMXv = Xv * V - bd["y0"][0] * bd["v_v"][0]
        if len(t) > 1:
            dt = np.diff(t)
            J = np.concatenate([[0.0], np.cumsum(0.5 * (Xv[:-1]*V[:-1] + Xv[1:]*V[1:]) * dt)])
            K = np.concatenate([[0.0], np.cumsum(0.5 * (Xv[:-1]*V[:-1]*Pv[:-1]/(0.5+Pv[:-1]) +
                                                         Xv[1:]*V[1:]*Pv[1:]/(0.5+Pv[1:])) * dt)])
        else:
            J = K = np.zeros(1)
        R.loc[idx, "J"] = J
        R.loc[idx, "dMXv"] = dMXv
        R.loc[idx, "K"] = K
        R.loc[idx, ["cumF_fs", "cumF_fb", "cumF_fc", "cumF_fd"]] = cumF
    return R


def balance_ols(Rtr):
    """Fit substrate balance: S*V - S0*V0 = g . cumF - a*Jmu - b*J  (OLS)
       and byproduct: P*V - P0*V0 = alpha*Jmu + beta*J - q*K (OLS)."""
    yS = (Rtr["substrate"] * Rtr["V"] - Rtr["S0"] * Rtr["V0"]).values
    XS = np.column_stack([Rtr["cumF_fs"], Rtr["cumF_fb"], Rtr["cumF_fc"], Rtr["cumF_fd"],
                          -Rtr["dMXv"], -Rtr["J"]])
    gS, *_ = np.linalg.lstsq(XS, yS, rcond=None)
    yP = (Rtr["byproduct"] * Rtr["V"] - Rtr["P0"] * Rtr["V0"]).values
    XP = np.column_stack([Rtr["dMXv"], Rtr["J"], -Rtr["K"]])
    gP, *_ = np.linalg.lstsq(XP, yP, rcond=None)
    return gS, gP


def add_phys(R, gS, gP):
    R = R.copy()
    cum_mass = gS[0]*R["cumF_fs"] + gS[1]*R["cumF_fb"] + gS[2]*R["cumF_fc"] + gS[3]*R["cumF_fd"]
    R["S_phys"] = (R["S0"]*R["V0"] + cum_mass - gS[4]*R["dMXv"] - gS[5]*R["J"]) / R["V"]
    R["P_phys"] = (R["P0"]*R["V0"] + gP[0]*R["dMXv"] + gP[1]*R["J"] + gP[2]*R["K"]) / R["V"]
    return R


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
    R = R.sort_values(["batch_id", "t"]).reset_index(drop=True)
    feats = list(BASE_FEATS) + (ODE_FEATS if not args.no_ode else []) + KNN_FEATS
    print(f"rows {len(R)}", flush=True)

    # step 1: vcd and P ensembles (OOF for train, full for hidden)
    print("step1: vcd ensemble OOF...", flush=True)
    vcd_pred, wv, mn = oof_predict(R, B, "vcd", feats, args.folds, not args.no_ode)
    print("step1b: byproduct first-pass OOF...", flush=True)
    P_pred, wp, _ = oof_predict(R, B, "byproduct", feats, args.folds, not args.no_ode)

    R["casc_vcd"] = vcd_pred
    R["casc_P"] = P_pred
    vcd_map = {b: R.loc[R.batch_id == b, "casc_vcd"].values for b in R.batch_id.unique()}
    P_map = {b: R.loc[R.batch_id == b, "casc_P"].values for b in R.batch_id.unique()}

    print("step2: cascade features...", flush=True)
    R = cascade_feats(R, B, vcd_map, P_map)
    CASC = ["J", "dMXv", "K", "cumF_fs", "cumF_fb", "cumF_fc", "cumF_fd", "casc_vcd", "casc_P"]

    # step 3: CV for S and P with phys features (per-fold OLS gains)
    print("step3: S/P CV...", flush=True)
    train = R[~R.is_tp].reset_index(drop=True)
    gkf = GroupKFold(n_splits=args.folds)
    model_names = list(make_models(0).keys())
    feats_all = feats + CASC + ["S_phys", "P_phys"]
    base_oof2 = {t: {m: np.zeros(len(train)) for m in model_names} for t in ("substrate", "byproduct")}
    for tr_idx, te_idx in gkf.split(train, groups=train.batch_id.values):
        tr = train.iloc[tr_idx].copy()
        te = train.iloc[te_idx].copy()
        ref_ids = sorted(tr.batch_id.unique(), key=lambda s: int(s[5:]))
        for b in ref_ids:
            add_knn_feats(tr, B, tr.batch_id == b, [x for x in ref_ids if x != b], k=5)
        add_knn_feats(te, B, te.index == te.index, ref_ids, k=5)
        gS, gP = balance_ols(tr)
        tr = add_phys(tr, gS, gP)
        te = add_phys(te, gS, gP)
        for t in ("substrate", "byproduct"):
            for name in model_names:
                mdl = LogWrap(make_models(0)[name])
                mdl.fit(tr[feats_all], tr[t])
                base_oof2[t][name][te_idx] = mdl.predict(te[feats_all])
    cv = {}
    for t in ("substrate", "byproduct"):
        y = train[t].values
        X = np.column_stack([base_oof2[t][m] for m in model_names])
        w, _ = nnls(np.column_stack([X, np.ones(len(X))]), y)
        oof = X @ w[:-1] + w[-1]
        mape = (np.abs(oof - y) / y).mean() * 100
        bm = {m: (np.abs(base_oof2[t][m] - y) / y).mean() * 100 for m in model_names}
        cv[t] = mape
        print(f"{t:10s} stack {mape:6.2f} | " + " ".join(f"{m}:{v:.2f}" for m, v in bm.items()), flush=True)
    print("S/P CV mean:", round(np.mean(list(cv.values())), 3), flush=True)


if __name__ == "__main__":
    main()

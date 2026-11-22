"""Statistical stacking pipeline for the DTB leaderboard.

Rows = (batch, time) for the 14 obs rows of the 49 observed batches and the
13 to-predict rows of the 12 hidden batches. Rich per-row features (initial
state, feed schedule, cumulative/recent feeds, volume, obs index, ODE-sim
trajectory values, cohort kNN trajectory) + per-target base models + NNLS/
Ridge stacking meta-learner. Honest CV = GroupKFold by batch.

Usage:
  python pipeline.py --cv                 # print CV MAPE per target
  python pipeline.py --submit out.csv     # write submission for hidden batches
"""
import sys, os, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import load_batches, SimSet, PNAMES

from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import nnls

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]


# ---------------------------------------------------------------- features
def batch_schedule_feats(bd):
    ft, fa = bd["f_t"], bd["f_a"]
    n = len(ft)
    tot = fa.sum(0) if n else np.zeros(4)
    cnt = (fa > 0).sum(0) if n else np.zeros(4)
    hist = []
    for c in range(4):
        h, _ = np.histogram(ft[fa[:, c] > 0], bins=np.arange(0, 15, 1.0),
                            weights=fa[fa[:, c] > 0, c])
        hist.append(h)
    hist = np.concatenate(hist)
    first = np.array([ft[fa[:, c] > 0][0] if (fa[:, c] > 0).any() else -1.0 for c in range(4)])
    last = np.array([ft[fa[:, c] > 0][-1] if (fa[:, c] > 0).any() else -1.0 for c in range(4)])
    return np.concatenate([tot, cnt, first, last, hist])


def build_rows(B, tpl, ode_params=None, knn_traj=None):
    """Return DataFrame with one row per (batch, row-time)."""
    rows = []
    ode_sims = None
    if ode_params is not None:
        all_ids = sorted(B.keys(), key=lambda s: int(s[5:]))
        sim = SimSet(B, all_ids, dt=0.005)
        p2 = np.exp(np.array(ode_params))[None, :].repeat(len(all_ids), axis=0)
        M_all = sim.simulate(p2)
        ode_sims = {}
        for j, b in enumerate(all_ids):
            tg = np.arange(sim.n_steps + 1) * sim.dt
            ode_sims[b] = (tg, M_all[:, j, :] / sim.V[:, j, None])
    for b, bd in B.items():
        g_rows_t = np.concatenate([bd["t_obs"], bd["t_pred"] if bd["t_pred"] is not None else []])
        # rows: observed rows (with y) + to-predict rows
        times = []
        ys = []
        is_tp = []
        for t, y in zip(bd["t_obs"], bd["y_obs"]):
            times.append(t); ys.append(y); is_tp.append(False)
        if bd["t_pred"] is not None:
            for t in bd["t_pred"]:
                times.append(t); ys.append(np.full(4, np.nan)); is_tp.append(True)
        order = np.argsort(times)
        times = np.array(times)[order]; ys = np.array(ys)[order]; is_tp = np.array(is_tp)[order]
        # volume at each time: interp from v_t/v_v
        V = np.interp(times, bd["v_t"], bd["v_v"])
        ft, fa = bd["f_t"], bd["f_a"]
        sch = batch_schedule_feats(bd)
        y0 = bd["y0"]
        for k, (t, y, tp) in enumerate(zip(times, ys, is_tp)):
            m = ft <= t + 1e-9
            cum = fa[m].sum(0) if m.any() else np.zeros(4)
            last1 = fa[(ft > t - 1.0) & (ft <= t + 1e-9)].sum(0)
            last2 = fa[(ft > t - 2.0) & (ft <= t + 1e-9)].sum(0)
            since = np.array([t - ft[fa[:, c] > 0][-1] if (fa[:, c] > 0).any() else 99.0 for c in range(4)])
            nxt = np.array([ft[fa[:, c] > 0][ft[fa[:, c] > 0] > t + 1e-9][0] - t
                            if ((fa[:, c] > 0) & (ft > t + 1e-9)).any() else 99.0 for c in range(4)])
            row = dict(batch_id=b, t=t, k=k, is_tp=tp, V=V[k],
                       vcd0=y0[0], xd0=y0[1], S0=y0[2], P0=y0[3], V0=bd["v_v"][0],
                       cum_fs=cum[0], cum_fb=cum[1], cum_fc=cum[2], cum_fd=cum[3],
                       last1_fs=last1[0], last1_fb=last1[1], last2_fb=last2[1],
                       since_fb=since[1], since_fs=since[0], nxt_fb=nxt[1], n_f_so_far=int(m.sum()),
                       vcd=y[0], tcd=y[0] + y[1], substrate=y[2], byproduct=y[3])
            row["sched"] = sch
            if ode_sims is not None:
                tg, tr = ode_sims[b]
                kk = int(round(t / 0.005))
                kk = min(kk, len(tg) - 1)
                row["ode_vcd"], row["ode_xd"], row["ode_S"], row["ode_P"] = tr[kk]
            if knn_traj is not None and b in knn_traj:
                row["knn_vcd"], row["knn_tcd"], row["knn_S"], row["knn_P"] = knn_traj[b][min(k, len(knn_traj[b]) - 1)]
            rows.append(row)
    R = pd.DataFrame(rows)
    sch = np.stack(R["sched"].values)
    R = R.drop(columns=["sched"])
    for i in range(sch.shape[1]):
        R[f"sch{i}"] = sch[:, i]
    return R


BASE_FEATS = ["t", "k", "V", "vcd0", "xd0", "S0", "P0", "V0",
              "cum_fs", "cum_fb", "cum_fc", "cum_fd",
              "last1_fs", "last1_fb", "last2_fb", "since_fb", "since_fs",
              "nxt_fb", "n_f_so_far"] + [f"sch{i}" for i in range(13 * 4 + 12)]
ODE_FEATS = ["ode_vcd", "ode_xd", "ode_S", "ode_P"]
KNN_FEATS = ["knn_vcd", "knn_tcd", "knn_S", "knn_P"]


# ---------------------------------------------------------------- models
def make_models(seed=0):
    return {
        "hgb": HistGradientBoostingRegressor(max_iter=1200, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=5, l2_regularization=1.0,
                                             random_state=seed),
        "hgb_deep": HistGradientBoostingRegressor(max_iter=800, learning_rate=0.04, max_leaf_nodes=63,
                                                  min_samples_leaf=4, l2_regularization=0.5,
                                                  random_state=seed + 1),
        "et": ExtraTreesRegressor(n_estimators=600, min_samples_leaf=3, max_features=0.7,
                                  random_state=seed, n_jobs=8),
        "rf": RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.7,
                                    random_state=seed, n_jobs=8),
        "knn": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=8, weights="distance")),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(0.1, 1000, 20))),
    }


def knn_traj_feature(B, ref_ids, target_ids, k=5):
    """kNN-average obs-index trajectory: for each TARGET batch, average over REF
    batches only (never the target itself -> no leakage)."""
    F = np.stack([np.concatenate([B[b]["y0"], B[b]["v_v"][:1], batch_schedule_feats(B[b])])
                  for b in ref_ids])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fn = (F - mu) / sd
    Obs = {b: B[b]["y_obs"] for b in ref_ids}
    out = {}
    for b in target_ids:
        f = np.concatenate([B[b]["y0"], B[b]["v_v"][:1], batch_schedule_feats(B[b])])
        d = ((Fn - (f - mu) / sd) ** 2).sum(1)
        order = np.argsort(d)[:k]
        w = 1.0 / (d[order] + 1e-9); w /= w.sum()
        traj = sum(w[i] * np.column_stack([Obs[ref_ids[j]][:, 0],
                                           Obs[ref_ids[j]][:, 0] + Obs[ref_ids[j]][:, 1],
                                           Obs[ref_ids[j]][:, 2], Obs[ref_ids[j]][:, 3]])
                   for i, j in enumerate(order))
        out[b] = traj
    return out


def add_knn_feats(R, B, target_mask, ref_ids, k=5):
    """Fill KNN_FEATS columns for rows whose batch is in target_mask, using ref_ids."""
    tgt_ids = sorted(R.loc[target_mask, "batch_id"].unique(), key=lambda s: int(s[5:]))
    traj = knn_traj_feature(B, ref_ids, tgt_ids, k=k)
    for b, tr in traj.items():
        m = R.batch_id == b
        ks = R.loc[m, "k"].values
        for i, cname in enumerate(["knn_vcd", "knn_tcd", "knn_S", "knn_P"]):
            R.loc[m, cname] = [tr[min(kk, len(tr) - 1), i] for kk in ks]
    return R


def cv_evaluate(R, B, use_ode=True, use_knn=True, folds=6, verbose=True, knn_k=5):
    train = R[~R.is_tp].copy().reset_index(drop=True)
    feats = list(BASE_FEATS)
    if use_ode:
        feats += ODE_FEATS
    if use_knn:
        feats += KNN_FEATS
    gkf = GroupKFold(n_splits=folds)
    groups = train.batch_id.values
    model_names = list(make_models().keys())
    base_oof = {t: {m: np.zeros(len(train)) for m in model_names} for t in TARGETS}
    for tr_idx, te_idx in gkf.split(train, groups=groups):
        tr = train.iloc[tr_idx].copy()
        te = train.iloc[te_idx].copy()
        if use_knn:
            ref_ids = sorted(tr.batch_id.unique(), key=lambda s: int(s[5:]))
            for b in ref_ids:
                others = [x for x in ref_ids if x != b]
                add_knn_feats(tr, B, tr.batch_id == b, others, k=knn_k)
            add_knn_feats(te, B, te.index == te.index, ref_ids, k=knn_k)
        for t in TARGETS:
            for name in model_names:
                mdl = make_models()[name]
                mdl.fit(tr[feats], tr[t])
                base_oof[t][name][te_idx] = mdl.predict(te[feats])
    results = {}
    for t in TARGETS:
        X = np.column_stack([base_oof[t][m] for m in model_names])
        y = train[t].values
        w, _ = nnls(np.column_stack([X, np.ones(len(X))]), y)
        oof_pred = X @ w[:-1] + w[-1]
        mape = (np.abs(oof_pred - y) / np.abs(y)).mean() * 100
        base_mapes = {m: (np.abs(base_oof[t][m] - y) / np.abs(y)).mean() * 100 for m in model_names}
        results[t] = dict(stack_mape=mape, base=base_mapes, w=dict(zip(model_names + ["const"], w)))
        if verbose:
            print(f"{t:10s} stack MAPE {mape:6.2f} | " +
                  " ".join(f"{m}:{v:.2f}" for m, v in base_mapes.items()), flush=True)
    mean = np.mean([results[t]["stack_mape"] for t in TARGETS])
    if verbose:
        print(f"CV mean MAPE (stack): {mean:.3f}")
    return results, mean, base_oof, model_names, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--submit", default=None)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--no-ode", action="store_true")
    ap.add_argument("--no-knn", action="store_true")
    ap.add_argument("--ode-params", default=os.path.join(os.path.dirname(__file__), "params_ms.json"))
    args = ap.parse_args()

    B, tpl, df = load_batches()
    train_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    hid_ids = sorted([b for b in B if B[b]["hidden"]], key=lambda s: int(s[5:]))

    ode_params = None
    if not args.no_ode and os.path.exists(args.ode_params):
        ode_params = json.load(open(args.ode_params))["logp"]
    R = build_rows(B, tpl, ode_params=ode_params, knn_traj=None)
    for c in KNN_FEATS:
        R[c] = 0.0
    print(f"rows: {len(R)}, train {int((~R.is_tp).sum())}, hidden {int(R.is_tp.sum())}")

    if args.cv:
        results, mean, base_oof, model_names, feats = cv_evaluate(
            R, B, use_ode=not args.no_ode, use_knn=not args.no_knn, folds=args.folds)
        print("mean", round(mean, 3))
        json.dump({t: {"stack_mape": results[t]["stack_mape"],
                       "base": results[t]["base"], "w": {k: float(v) for k, v in results[t]["w"].items()}}
                   for t in TARGETS}, open(os.path.join(os.path.dirname(__file__), "cv_results.json"), "w"), indent=1)

    if args.submit:
        train = R[~R.is_tp].copy().reset_index(drop=True)
        hid = R[R.is_tp].copy().reset_index(drop=True)
        feats = list(BASE_FEATS)
        if not args.no_ode:
            feats += ODE_FEATS
        if not args.no_knn:
            feats += KNN_FEATS
            # LOO knn features for train rows over all 49; hidden rows over all 49
            for b in train_ids:
                others = [x for x in train_ids if x != b]
                add_knn_feats(train, B, train.batch_id == b, others, k=5)
            add_knn_feats(hid, B, hid.index == hid.index, train_ids, k=5)
        gkf = GroupKFold(n_splits=args.folds)
        model_names = list(make_models().keys())
        base_oof = {t: {m: np.zeros(len(train)) for m in model_names} for t in TARGETS}
        for tr_idx, te_idx in gkf.split(train, groups=train.batch_id.values):
            tr = train.iloc[tr_idx].copy()
            te = train.iloc[te_idx].copy()
            if not args.no_knn:
                ref_ids = sorted(tr.batch_id.unique(), key=lambda s: int(s[5:]))
                for b in ref_ids:
                    others = [x for x in ref_ids if x != b]
                    add_knn_feats(tr, B, tr.batch_id == b, others, k=5)
                add_knn_feats(te, B, te.index == te.index, ref_ids, k=5)
            for t in TARGETS:
                for name in model_names:
                    mdl = make_models()[name]
                    mdl.fit(tr[feats], tr[t])
                    base_oof[t][name][te_idx] = mdl.predict(te[feats])
        out = hid[["batch_id", "t"]].copy()
        for t in TARGETS:
            Xtr = np.column_stack([base_oof[t][m] for m in model_names])
            w, _ = nnls(np.column_stack([Xtr, np.ones(len(Xtr))]), train[t].values)
            Xh = []
            for name in model_names:
                mdl = make_models()[name]
                mdl.fit(train[feats], train[t])
                Xh.append(mdl.predict(hid[feats]))
            Xh = np.column_stack(Xh)
            pred = Xh @ w[:-1] + w[-1]
            out[t] = np.maximum(pred, 0.0)
            print(f"{t}: nnls weights", dict(zip(model_names + ["const"], np.round(w, 3))), flush=True)
        out = out.rename(columns={"t": "elapsed_time"})
        out = out[["batch_id", "elapsed_time"] + TARGETS]
        out.to_csv(args.submit, index=False)
        print("wrote", args.submit)


if __name__ == "__main__":
    main()

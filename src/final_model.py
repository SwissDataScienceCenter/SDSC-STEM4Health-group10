"""Final model: per-target ensemble of simple ML models + cohort shrinkage.

Members per target:
  - HistGradientBoosting on row features (t, initial state, feed schedule,
    cumulative feeds, V)  [+ cascade features (predicted vcd, biomass integral)
    for substrate/byproduct]
  - ExtraTrees on the same features
  - kNN trajectory match (k nearest train batches by initial state + feed
    schedule, average their observation-index trajectories)

Per-target shrinkage toward the cohort-median trajectory, weight fit on a
holdout set of batches (honest validation, MAPE on the 13 future points).
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
DATA_CANDIDATES = [
    "/home/renku/work/input-data/Project 1 - DTB",
    os.path.join(os.path.dirname(__file__), "..", "..", "input-data", "Project 1 - DTB"),
    "./data", ".",
]


def find_data_dir():
    for c in DATA_CANDIDATES:
        if os.path.exists(os.path.join(c, "bioreaction_remaining.csv")):
            return os.path.abspath(c)
    raise FileNotFoundError(DATA_CANDIDATES)


def load_batches(df):
    """Per batch: dict with y0, V0, feed arrays, obs matrix (14x4), obs times, hidden flag."""
    B = {}
    for b, g in df.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        o = g[g.vcd.notna()]
        f = g[g[FEEDS].notna().any(axis=1)]
        y0r = o.iloc[0]
        B[str(b)] = dict(
            hidden=bool(g.to_predict.any()),
            y0=np.array([y0r.vcd, y0r.tcd - y0r.vcd, y0r.substrate, y0r.byproduct], float),
            V0=float(y0r.volume),
            obs=o[["vcd", "tcd", "substrate", "byproduct"]].values.astype(float) if len(o) > 1 else None,
            t_obs=o.elapsed_time.values.astype(float) if len(o) > 1 else np.array([float(y0r.elapsed_time)]),
            f_t=f.elapsed_time.values.astype(float),
            f_a=f[FEEDS].fillna(0.0).values.astype(float),
            V_at=g.set_index("elapsed_time")["volume"],
        )
    return B


def batch_features(bd):
    ft, fa = bd["f_t"], bd["f_a"]
    tot = fa.sum(0) if len(fa) else np.zeros(4)
    return np.array([
        bd["y0"][0], bd["y0"][1], bd["y0"][2], bd["y0"][3], bd["V0"],
        len(ft), tot[0], tot[1], tot[2], tot[3],
        ft[fa[:, 1] > 0][0] if (fa[:, 1] > 0).any() else 0.0,
        np.diff(np.sort(ft)).mean() if len(ft) > 1 else 0.0,
    ], float)


def row_features(bd, t):
    ft, fa = bd["f_t"], bd["f_a"]
    m = ft <= t + 1e-9
    cum = fa[m].sum(0) if m.any() else np.zeros(4)
    c1 = fa[(ft > t - 1.0) & (ft <= t + 1e-9)].sum(0) if m.any() else np.zeros(4)
    last_fb = ft[fa[:, 1] > 0][-1] if (fa[:, 1] > 0).any() else 0.0
    bf = batch_features(bd)
    V = bd["V_at"].get(t, np.nan)
    if not np.isfinite(V):
        V = bd["V0"] + cum.sum()
    return np.r_[bf, t, cum, c1, len(ft[m]), t - last_fb, V]


FEAT_NAMES = ["vcd0", "xd0", "S0", "P0", "V0", "n_f", "tot_fs", "tot_fb", "tot_fc", "tot_fd",
              "first_fb", "mean_dt_f", "t", "cum_fs", "cum_fb", "cum_fc", "cum_fd",
              "cum_1d_fs", "cum_1d_fb", "cum_1d_fc", "cum_1d_fd", "n_f_so_far", "t_since_fb", "V"]


def build_train_matrix(B, ids):
    X, Y, bid, idx = [], [], [], []
    for b in ids:
        bd = B[b]
        if bd["obs"] is None:
            continue
        for k, t in enumerate(bd["t_obs"]):
            X.append(row_features(bd, t))
            Y.append(bd["obs"][k])
            bid.append(b); idx.append(k)
    return np.array(X), np.array(Y), np.array(bid), np.array(idx)


def knn_traj(B, ids, k=10):
    F = np.stack([batch_features(B[b]) for b in ids])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fn = (F - mu) / sd
    O = {b: B[b]["obs"] for b in ids if B[b]["obs"] is not None}
    return F, Fn, O, mu, sd


def knn_predict(b, F, Fn, O, ids, mu, sd, k=10):
    f = (batch_features(B_global[b]) - mu) / sd
    d = ((Fn - f) ** 2).sum(1)
    nn = [ids[i] for i in np.argsort(d)[:k] if ids[i] in O]
    return np.mean([O[i] for i in nn], axis=0)  # (14,4)


def gbm():
    return HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.06,
                                         max_leaf_nodes=63, min_samples_leaf=5,
                                         l2_regularization=1.0, random_state=0)


def etree():
    return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=4, random_state=0, n_jobs=1)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--holdout-every", type=int, default=8)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    global B_global
    data_dir = find_data_dir()
    df = pd.read_csv(os.path.join(data_dir, "bioreaction_remaining.csv"))
    tpl = pd.read_csv(os.path.join(data_dir, "bioreaction_template.csv"), dtype={"batch_id": str})
    B = load_batches(df)
    B_global = B
    train_ids = sorted([b for b in B if B[b]["obs"] is not None], key=lambda s: int(s[5:]))
    pred_ids = sorted([b for b in B if B[b]["hidden"]], key=lambda s: int(s[5:]))
    print(f"train: {len(train_ids)} batches, hidden: {len(pred_ids)}", flush=True)

    # ---------------- validation ----------------
    holdout = train_ids[::args.holdout_every]
    tr_ids = [b for b in train_ids if b not in holdout]
    Xtr, Ytr, bid_tr, idx_tr = build_train_matrix(B, tr_ids)
    Xho, Yho, bid_ho, idx_ho = build_train_matrix(B, holdout)

    med_traj = np.median(np.stack([B[b]["obs"] for b in tr_ids]), axis=0)  # (14,4)

    # cascade: predicted vcd + biomass integral as extra features for S/P
    m_vcd_v = gbm().fit(Xtr, Ytr[:, 0])
    vcd_hat_tr = m_vcd_v.predict(Xtr)
    vcd_hat_ho = m_vcd_v.predict(Xho)
    # simpler: compute int_vcd with a loop
    def add_cascade(X, bid, vhat):
        t = X[:, FEAT_NAMES.index("t")]
        iv = np.zeros(len(X))
        order = np.lexsort((t, bid))
        # group by batch
        for b in np.unique(bid):
            m = bid == b
            tt, vv = t[m], vhat[m]
            o = np.argsort(tt)
            iv[m][o] = np.concatenate([[0.0], np.cumsum(0.5 * (vv[o][1:] + vv[o][:-1]) * np.diff(tt[o]))]) if len(tt) > 1 else [0.0]
        return np.column_stack([X, vhat, iv])

    Xtr2 = add_cascade(Xtr, bid_tr, vcd_hat_tr)
    Xho2 = add_cascade(Xho, bid_ho, vcd_hat_ho)
    F2N = FEAT_NAMES + ["vcd_hat", "int_vcd"]

    F_knn, Fn_knn, O_knn, mu_f, sd_f = knn_traj(B, tr_ids)

    val = {}
    for c, t in enumerate(TARGETS):
        feats = F2N if t in ("substrate", "byproduct") else FEAT_NAMES
        Xt = Xtr2 if t in ("substrate", "byproduct") else Xtr
        Xh = Xho2 if t in ("substrate", "byproduct") else Xho
        m1 = gbm().fit(Xt, Ytr[:, c])
        m2 = etree().fit(Xt, Ytr[:, c])
        p_gbm = m1.predict(Xh); p_et = m2.predict(Xh)
        # knn per holdout batch
        p_knn = np.zeros(len(Xh))
        for b in holdout:
            m = bid_ho == b
            traj = knn_predict(b, F_knn, Fn_knn, O_knn, tr_ids, mu_f, sd_f, k=args.knn_k)
            for j, i in enumerate(np.where(m)[0]):
                p_knn[i] = traj[idx_ho[i], c]
        ens = (p_gbm + p_et) / 2
        blend_knn = 0.5 * ens + 0.5 * p_knn
        # median traj per row
        p_med = np.array([med_traj[i, c] for i in idx_ho])
        # grid-search shrinkage toward median
        best_w, best_m = 1.0, np.inf
        for w in np.arange(0, 1.01, 0.1):
            pr = w * blend_knn + (1 - w) * p_med
            m_ = (np.abs(pr - Yho[:, c]) / Yho[:, c])
            m_future = m_[idx_ho > 0].mean() * 100
            if m_future < best_m:
                best_m, best_w = m_future, w
        val[t] = (best_w, best_m)
        print(f"holdout {t:10s}: w_med={best_w:.1f}  MAPE={best_m:5.2f}%   (gbm-only future MAPE: "
              f"{(np.abs(p_gbm[idx_ho>0]-Yho[idx_ho>0,c])/Yho[idx_ho>0,c]).mean()*100:.2f}%)", flush=True)
    print(f"holdout mean MAPE: {np.mean([v[1] for v in val.values()]):.2f}%", flush=True)

    # ---------------- final fit on all train, predict hidden ----------------
    Xa, Ya, bid_a, idx_a = build_train_matrix(B, train_ids)
    m_vcd = gbm().fit(Xa, Ya[:, 0])
    vcd_hat_a = m_vcd.predict(Xa)
    Xa2 = add_cascade(Xa, bid_a, vcd_hat_a)

    med_all = np.median(np.stack([B[b]["obs"] for b in train_ids]), axis=0)
    F_knn, Fn_knn, O_knn, mu_f, sd_f = knn_traj(B, train_ids)

    tpl_times = {b: tpl[tpl.batch_id == b].sort_values("elapsed_time").elapsed_time.values.astype(float)
                 for b in pred_ids}
    models = {}
    for c, tc in enumerate(TARGETS):
        use2 = tc in ("substrate", "byproduct")
        Xuse = Xa2 if use2 else Xa
        models[tc] = (gbm().fit(Xuse, Ya[:, c]), etree().fit(Xuse, Ya[:, c]))

    rows = []
    for b in pred_ids:
        bd = B[b]
        Xh = np.stack([row_features(bd, t) for t in tpl_times[b]])
        vcd_hat_h = m_vcd.predict(Xh)
        Xh2 = add_cascade(Xh, np.array([b] * len(Xh)), vcd_hat_h)
        traj = knn_predict(b, F_knn, Fn_knn, O_knn, train_ids, mu_f, sd_f, k=args.knn_k)
        for j, t in enumerate(tpl_times[b]):   # template rows = future points -> obs index j+1
            row = {"batch_id": b, "elapsed_time": t}
            for c, tc in enumerate(TARGETS):
                Xuse = Xh2 if tc in ("substrate", "byproduct") else Xh
                m1, m2 = models[tc]
                ens = (m1.predict(Xuse[j:j+1]) + m2.predict(Xuse[j:j+1])) / 2
                knn_v = traj[min(j + 1, 13), c]
                blend = 0.5 * ens[0] + 0.5 * knn_v
                w = val[tc][0]
                pred = w * blend + (1 - w) * med_all[min(j + 1, 13), c]
                row[tc] = max(pred, 0.0)
            rows.append(row)
    fill = pd.DataFrame(rows)
    tpl["_k"] = tpl.batch_id + "|" + tpl.elapsed_time.round(6).astype(str)
    fill["_k"] = fill.batch_id + "|" + fill.elapsed_time.round(6).astype(str)
    out = tpl.drop(columns=TARGETS).merge(fill, on=["_k", "batch_id", "elapsed_time"], how="left")
    sub = out[["batch_id", "elapsed_time"] + TARGETS]
    assert sub.notna().all().all() and np.isfinite(sub[TARGETS].values).all()
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nwrote {os.path.abspath(out_path)} shape={sub.shape}", flush=True)
    print(sub.head(3).to_string())


if __name__ == "__main__":
    main()

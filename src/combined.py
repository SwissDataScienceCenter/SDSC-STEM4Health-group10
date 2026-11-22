"""Combined simple model: per-target GBM (row features) + kNN trajectory + median shrinkage.

Honest validation on holdout batches (MAPE over the 13 future points), then
final fit on all 49 batches and prediction of the 12 hidden batches.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from simple_ml import build_rows, find_data_dir, FEATS, TARGETS, FEEDS


def gbm():
    return HistGradientBoostingRegressor(max_iter=800, learning_rate=0.07,
                                         max_leaf_nodes=31, min_samples_leaf=6,
                                         l2_regularization=1.0, random_state=0)


def batch_feat_vec(df, b):
    g = df[df.batch_id == b].sort_values("elapsed_time")
    o = g[g.vcd.notna()]
    f = g[g[FEEDS].notna().any(axis=1)]
    fa = f[FEEDS].fillna(0.0).values
    ft = f.elapsed_time.values
    y0 = o.iloc[0]
    return np.array([y0.vcd, y0.tcd - y0.vcd, y0.substrate, y0.byproduct, y0.volume,
                     len(ft), fa[:, 0].sum(), fa[:, 1].sum(), fa[:, 2].sum(), fa[:, 3].sum(),
                     ft[fa[:, 1] > 0][0] if (fa[:, 1] > 0).any() else 0.0], float)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--holdout-every", type=int, default=8)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data_dir = find_data_dir()
    df = pd.read_csv(os.path.join(data_dir, "bioreaction_remaining.csv"))
    tpl = pd.read_csv(os.path.join(data_dir, "bioreaction_template.csv"), dtype={"batch_id": str})
    R = build_rows(df)
    pred_ids = sorted(R[R.to_predict].batch_id.unique(), key=lambda s: int(s[5:]))
    train = R[~R.batch_id.isin(pred_ids)].copy()
    hid = R[R.batch_id.isin(pred_ids) & R.to_predict].copy()
    train_ids = sorted(train.batch_id.unique(), key=lambda s: int(s[5:]))
    for D in (train, hid):
        m = D.V.isna()
        D.loc[m, "V"] = D.loc[m, "V0"] + D.loc[m, ["cum_fb", "cum_fs", "cum_fc", "cum_fd"]].sum(axis=1)

    # kNN structures
    Fb = {b: batch_feat_vec(df, b) for b in train_ids}
    F = np.stack([Fb[b] for b in train_ids])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fn = (F - mu) / sd
    Obs = {b: train[train.batch_id == b].sort_values("elapsed_time")[TARGETS].values for b in train_ids}

    def knn_traj(b, k):
        f = (batch_feat_vec(df, b) - mu) / sd
        d = ((Fn - f) ** 2).sum(1)
        nn = [train_ids[i] for i in np.argsort(d)[:k]]
        return np.mean([Obs[i] for i in nn], axis=0)  # (14,4)

    holdout = train_ids[::args.holdout_every]
    tr = train[~train.batch_id.isin(holdout)]
    ho = train[train.batch_id.isin(holdout)]
    med = np.median(np.stack([Obs[b] for b in train_ids if b not in holdout]), axis=0)

    # cascade features for byproduct: predicted vcd and its cumulative integral
    m_vcd_c = gbm().fit(tr[FEATS], tr["vcd"])
    def cascade(D, feats):
        v = m_vcd_c.predict(D[feats])
        tcol = D["elapsed_time"].values
        bid = D["batch_id"].values
        iv = np.zeros(len(D))
        for b in np.unique(bid):
            m = bid == b
            tt, vv = tcol[m], v[m]
            o = np.argsort(tt)
            iv[m] = 0.0
            iv[m][o] = np.concatenate([[0.0], np.cumsum(0.5*(vv[o][1:]+vv[o][:-1])*np.diff(tt[o]))]) if len(tt) > 1 else [0.0]
        return np.column_stack([D[feats].values, v, iv])
    FEATS_C = FEATS + ["vcd_hat", "int_vcd"]

    val = {}
    for c, t in enumerate(TARGETS):
        use_c = (t == "byproduct")
        Xt = cascade(tr, FEATS) if use_c else tr[FEATS]
        Xh = cascade(ho, FEATS) if use_c else ho[FEATS]
        m = gbm().fit(Xt, tr[t])
        p_gbm = m.predict(Xh)
        p_knn = np.zeros(len(ho))
        ho_s = ho.sort_values(["batch_id", "elapsed_time"])
        for b in holdout:
            traj = knn_traj(b, args.knn_k)
            rows_b = ho[ho.batch_id == b].sort_values("elapsed_time")
            for j, ii in enumerate(rows_b.index):
                p_knn[ho.index.get_loc(ii)] = traj[min(j, 13), c]
        p_med = np.zeros(len(ho))
        for b in holdout:
            rows_b = ho[ho.batch_id == b].sort_values("elapsed_time")
            for j, ii in enumerate(rows_b.index):
                p_med[ho.index.get_loc(ii)] = med[min(j, 13), c]
        fut = ho.elapsed_time > 0
        y = ho.loc[fut, t].values
        members = {"gbm": p_gbm[fut.values], "knn": p_knn[fut.values], "med": p_med[fut.values]}
        best_w, best_m = {"gbm": 1.0, "knn": 0.0, "med": 0.0}, np.inf
        for wg in np.arange(0, 1.01, 0.25):
            for wk in np.arange(0, 1.01 - wg, 0.25):
                wm = 1 - wg - wk
                pr = wg * members["gbm"] + wk * members["knn"] + wm * members["med"]
                mm = (np.abs(pr - y) / y).mean() * 100
                if mm < best_m:
                    best_m, best_w = mm, {"gbm": wg, "knn": wk, "med": wm}
        val[t] = (best_w, best_m, m, use_c)
        print(f"holdout {t:10s}: weights gbm={best_w['gbm']:.2f} knn={best_w['knn']:.2f} med={best_w['med']:.2f} -> {best_m:5.2f}%", flush=True)
    print(f"holdout mean MAPE: {np.mean([v[1] for v in val.values()]):.2f}%", flush=True)

    # ---------- final ----------
    m_vcd_f = gbm().fit(train[FEATS], train["vcd"])
    def cascade_f(D):
        v = m_vcd_f.predict(D[FEATS])
        tcol = D["elapsed_time"].values; bid = D["batch_id"].values
        iv = np.zeros(len(D))
        for b in np.unique(bid):
            m = bid == b; tt, vv = tcol[m], v[m]; o = np.argsort(tt)
            iv[m] = 0.0
            iv[m][o] = np.concatenate([[0.0], np.cumsum(0.5*(vv[o][1:]+vv[o][:-1])*np.diff(tt[o]))]) if len(tt) > 1 else [0.0]
        return np.column_stack([D[FEATS].values, v, iv])
    Xa_c = cascade_f(train)
    models = {}
    for t in TARGETS:
        models[t] = gbm().fit(Xa_c if val[t][3] else train[FEATS], train[t])
    med_all = np.median(np.stack([Obs[b] for b in train_ids]), axis=0)
    F = np.stack([Fb[b] for b in train_ids]); mu, sd = F.mean(0), F.std(0) + 1e-9
    Fn = (F - mu) / sd

    hid_s = hid.sort_values(["batch_id", "elapsed_time"])
    preds = {t: np.zeros(len(hid_s)) for t in TARGETS}
    for b in pred_ids:
        rows_b = hid_s[hid_s.batch_id == b]
        traj = knn_traj(b, args.knn_k)
        Xb = cascade_f(rows_b) if val[t][3] else rows_b[FEATS]
        for c, t in enumerate(TARGETS):
            Xb = cascade_f(rows_b) if val[t][3] else rows_b[FEATS]
            p_gbm = models[t].predict(Xb)
            w = val[t][0]
            for j, ii in enumerate(rows_b.index):
                knn_v = traj[min(j + 1, 13), c]   # hidden rows start at index 1
                pr = w["gbm"]*p_gbm[j] + w["knn"]*knn_v + w["med"]*med_all[min(j+1,13), c]
                preds[t][hid_s.index.get_loc(ii)] = max(pr, 0.0)
    hid_s = hid_s.copy()
    for t in TARGETS:
        hid_s[t] = preds[t]

    tpl["_k"] = tpl.batch_id + "|" + tpl.elapsed_time.round(6).astype(str)
    hid_s["_k"] = hid_s.batch_id + "|" + hid_s.elapsed_time.round(6).astype(str)
    out = tpl.drop(columns=TARGETS).merge(hid_s[["_k"] + TARGETS], on="_k", how="left")
    sub = out[["batch_id", "elapsed_time"] + TARGETS]
    assert sub.notna().all().all() and np.isfinite(sub[TARGETS].values).all()
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nwrote {os.path.abspath(out_path)} shape={sub.shape}", flush=True)
    print(sub.head(3).to_string())


if __name__ == "__main__":
    main()

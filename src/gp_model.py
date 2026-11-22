"""Gaussian process model for the fed-batch challenge.

Design: one GP per target over the joint input x = (standardized batch
features, time t, volume/cumulative feeds at t), with an anisotropic
(ARD) RBF kernel + constant + white noise:
    k(x, x') = c^2 * exp(-1/2 sum_d (x_d - x'_d)^2 / l_d^2) + noise
ARD length scales learn which inputs matter; the time dimension makes the GP
share information across days (a smooth trajectory model per "batch type").

Variants:
  --mode joint   : 4 GPs over (features, t)                       [default]
  --mode day     : one GP per (target, day-index) over batch features only
  --blend-knn W  : blend GP predictions with weighted-kNN (W = GP weight)

Honest validation: holdout batches, MAPE over the 13 future points.
"""
import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from simple_ml import build_rows, find_data_dir, TARGETS, FEEDS

FEEDS_R = FEEDS
def _extend_feats(R, df):
    # add feed-recency features per row
    extra = []
    fcache = {}
    for b, g in df.groupby("batch_id"):
        f = g[g[FEEDS].notna().any(axis=1)]
        fcache[b] = (f.elapsed_time.values.astype(float), f[FEEDS].fillna(0.0).values.astype(float))
    for _, r in R.iterrows():
        ft, fa = fcache[r.batch_id]
        t = r.elapsed_time
        c1 = fa[(ft > t - 1.0) & (ft <= t + 1e-9)].sum(0) if len(ft) else np.zeros(4)
        c05 = fa[(ft > t - 0.5) & (ft <= t + 1e-9)].sum(0) if len(ft) else np.zeros(4)
        last_fb = ft[fa[:, 1] > 0][-1] if (fa[:, 1] > 0).any() else t
        last_fs = ft[fa[:, 0] > 0][-1] if (fa[:, 0] > 0).any() else t
        extra.append([c1[0], c1[1], c05[0], c05[1], t - last_fb, t - last_fs])
    E = pd.DataFrame(extra, index=R.index, columns=["cum_fs_1d", "cum_fb_1d", "cum_fs_05d", "cum_fb_05d", "t_since_fb", "t_since_fs"])
    return pd.concat([R, E], axis=1)

FEATS = ["elapsed_time", "vcd0", "xd0", "S0", "P0", "V0", "n_f", "tot_fb", "tot_fs",
         "tot_fc", "tot_fd", "first_fb", "cum_fb", "cum_fs", "cum_fc", "cum_fd",
         "n_f_so_far", "V", "cum_fs_1d", "cum_fb_1d", "cum_fs_05d", "cum_fb_05d",
         "t_since_fb", "t_since_fs"]


def gp(n_dim, nu=1.5):
    kern = (ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=kern, alpha=1e-8, normalize_y=True,
                                    n_restarts_optimizer=1, random_state=0)


def knn_traj_preds(df, tpl, train_ids, pred_ids, k=15, power=2.0):
    from knn_model import batch_feats
    F = np.stack([batch_feats(df, b) for b in train_ids])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fn = (F - mu) / sd
    Obs = {b: df[(df.batch_id == b) & df.vcd.notna()].sort_values("elapsed_time")[TARGETS].values
           for b in train_ids}
    out = {}
    for b in pred_ids:
        d = ((Fn - (batch_feats(df, b) - mu) / sd) ** 2).sum(1)
        order = np.argsort(d)[:k]
        w = 1.0 / (d[order] ** power + 1e-9)
        w /= w.sum()
        traj = sum(w[j] * Obs[train_ids[i]] for j, i in enumerate(order))
        if b in set(tpl.batch_id):
            tt = tpl[tpl.batch_id == b].sort_values("elapsed_time").elapsed_time.values
            shift = 1
        else:  # holdout batch: use its own observed times (future points only)
            tt = df[(df.batch_id == b) & df.vcd.notna()].sort_values("elapsed_time").elapsed_time.values[1:]
            shift = 0
        out[b] = np.stack([np.maximum(traj[min(j + shift, 13)], 0.0) for j in range(len(tt))])
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["joint", "day"], default="joint")
    p.add_argument("--holdout-every", type=int, default=8)
    p.add_argument("--blend-knn", type=float, default=0.0, help="weight of GP in GP/kNN blend (0 = pure kNN)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data_dir = find_data_dir()
    df = pd.read_csv(os.path.join(data_dir, "bioreaction_remaining.csv"))
    tpl = pd.read_csv(os.path.join(data_dir, "bioreaction_template.csv"), dtype={"batch_id": str})
    R = _extend_feats(build_rows(df), df)
    pred_ids = sorted(R[R.to_predict].batch_id.unique(), key=lambda s: int(s[5:]))
    train = R[~R.batch_id.isin(pred_ids)].copy()
    hid = R[R.batch_id.isin(pred_ids) & R.to_predict].copy()
    for D in (train, hid):
        m = D.V.isna()
        D.loc[m, "V"] = D.loc[m, "V0"] + D.loc[m, ["cum_fb", "cum_fs", "cum_fc", "cum_fd"]].sum(axis=1)
    train_ids = sorted(train.batch_id.unique(), key=lambda s: int(s[5:]))
    print(f"train rows {len(train)}, hidden rows {len(hid)}, features {len(FEATS)}", flush=True)

    # standardize features with train stats
    mu, sd = train[FEATS].mean(), train[FEATS].std() + 1e-9

    def X_of(D):
        return ((D[FEATS] - mu) / sd).values

    holdout = train_ids[::args.holdout_every]
    tr = train[~train.batch_id.isin(holdout)]
    ho = train[train.batch_id.isin(holdout)]
    Xtr, Xho = X_of(tr), X_of(ho)
    fut = (ho.elapsed_time > 0).values

    gp_val = {}
    for c, t in enumerate(TARGETS):
        ytr, yho = tr[t].values, ho[t].values
        if args.mode == "joint":
            m = gp(len(FEATS)).fit(Xtr, ytr)
            p_ho = m.predict(Xho)
        else:
            # per day-index: separate GP per observation index
            p_ho = np.zeros(len(ho))
            idx_tr = tr.groupby("batch_id").cumcount().values
            idx_ho = ho.groupby("batch_id").cumcount().values
            # batch features only (drop time-varying cols for day mode)
            bcols = ["vcd0", "xd0", "S0", "P0", "V0", "n_f", "tot_fb", "tot_fs", "tot_fc", "tot_fd"]
            Xtr_b = ((tr[bcols] - mu[bcols]) / sd[bcols]).values
            Xho_b = ((ho[bcols] - mu[bcols]) / sd[bcols]).values
            for j in range(14):
                mtr = idx_tr == j
                if mtr.sum() < 5:
                    p_ho[idx_ho == j] = ytr[mtr].mean() if mtr.sum() else ytr.mean()
                    continue
                m = gp(len(bcols)).fit(Xtr_b[mtr], ytr[mtr])
                p_ho[idx_ho == j] = m.predict(Xho_b[idx_ho == j])
        mape = (np.abs(p_ho[fut] - yho[fut]) / yho[fut]).mean() * 100
        gp_val[t] = (mape, p_ho)
        print(f"holdout GP-{args.mode} {t:10s}: {mape:6.2f}%   kernel: "
              f"{str(m.kernel_)[:80] if args.mode=='joint' else ''}", flush=True)
    print(f"holdout GP mean MAPE: {np.mean([v[0] for v in gp_val.values()]):.2f}%", flush=True)

    # optional blend weight with kNN, evaluated on holdout
    w_gp = args.blend_knn
    if w_gp > 0:
        knn_ho = knn_traj_preds(df, tpl, [b for b in train_ids if b not in holdout], holdout)
        blend_w = {}
        for c, t in enumerate(TARGETS):
            p_knn = np.zeros(len(ho))
            for b in holdout:
                rows_b = ho[ho.batch_id == b].sort_values("elapsed_time")
                traj = knn_ho[b]
                for j, ii in enumerate(rows_b.index):
                    p_knn[ho.index.get_loc(ii)] = traj[j, c]
            best_w, best_m = 1.0, np.inf
            for w in np.arange(0, 1.01, 0.1):
                pr = w * gp_val[t][1] + (1 - w) * p_knn
                mm = (np.abs(pr[fut] - ho.loc[fut, t]) / ho.loc[fut, t]).mean() * 100
                if mm < best_m:
                    best_m, best_w = mm, w
            blend_w[t] = best_w
            print(f"  blend {t}: best GP weight {best_w:.1f} -> {best_m:.2f}%", flush=True)
        globals()["BLEND_W"] = blend_w

    # ---------- final: fit on all train, predict hidden ----------
    Xa = X_of(train)
    hid_X = X_of(hid)
    knn_all = knn_traj_preds(df, tpl, train_ids, pred_ids) if w_gp > 0 else None
    rows = []
    hid_s = hid.sort_values(["batch_id", "elapsed_time"])
    for c, t in enumerate(TARGETS):
        y = train[t].values
        if args.mode == "joint":
            m = gp(len(FEATS)).fit(Xa, y)
            p_hid = m.predict(X_of(hid_s))
            print(f"fit {t}: kernel {str(m.kernel_)[:100]}", flush=True)
        else:
            idx_a = train.groupby("batch_id").cumcount().values
            idx_h = hid_s.groupby("batch_id").cumcount().values + 1  # hidden rows are future points 1..13
            bcols = ["vcd0", "xd0", "S0", "P0", "V0", "n_f", "tot_fb", "tot_fs", "tot_fc", "tot_fd"]
            Xa_b = ((train[bcols] - mu[bcols]) / sd[bcols]).values
            Xh_b = ((hid_s[bcols] - mu[bcols]) / sd[bcols]).values
            p_hid = np.zeros(len(hid_s))
            for j in range(14):
                mtr = idx_a == j
                mh = idx_h == j
                if not mh.any():
                    continue
                if mtr.sum() < 5:
                    p_hid[mh] = y[mtr].mean() if mtr.sum() else y.mean()
                    continue
                mm = gp(len(bcols)).fit(Xa_b[mtr], y[mtr])
                p_hid[mh] = mm.predict(Xh_b[mh])
        gp_col = p_hid
        if w_gp > 0:
            p_knn = np.array([knn_all[b][j, c] for b in hid_s.batch_id for j in
                              range(len(hid_s[hid_s.batch_id == b]))])
            # align: hid_s sorted by batch/time; knn_all[b] rows in tpl time order = same order
            p_knn = np.zeros(len(hid_s))
            for b in pred_ids:
                rows_b = hid_s[hid_s.batch_id == b]
                for j, ii in enumerate(rows_b.index):
                    p_knn[hid_s.index.get_loc(ii)] = knn_all[b][j, c]
            bw = BLEND_W.get(t, w_gp) if w_gp > 0 else 0.0
            gp_col = bw * gp_col + (1 - bw) * p_knn
        hid_s = hid_s.copy()
        hid_s[t] = np.maximum(gp_col, 0.0)

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

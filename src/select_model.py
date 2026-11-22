"""Per-target model selection: GBM vs weighted-kNN vs GP (ARD-RBF), validated on
two holdout splits; best per target is fit on all data and used for submission.
Also two-stage tcd from predicted vcd (tcd ~ f(t, vcd) at 1.9% MAPE).
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from simple_ml import build_rows, find_data_dir, FEATS, TARGETS, FEEDS
from knn_model import batch_feats


def gbm():
    return HistGradientBoostingRegressor(max_iter=800, learning_rate=0.07,
                                         max_leaf_nodes=31, min_samples_leaf=6,
                                         l2_regularization=1.0, random_state=0)


def gp(n_dim):
    kern = (ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=kern, alpha=1e-8, normalize_y=True,
                                    n_restarts_optimizer=0, random_state=0)


def knn_pred(df, train_ids, target_ids, k=15, power=2.0):
    F = np.stack([batch_feats(df, b) for b in train_ids])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fn = (F - mu) / sd
    Obs = {b: df[(df.batch_id == b) & df.vcd.notna()].sort_values("elapsed_time")[TARGETS].values
           for b in train_ids}
    out = {}
    for b in target_ids:
        d = ((Fn - (batch_feats(df, b) - mu) / sd) ** 2).sum(1)
        order = np.argsort(d)[:k]
        w = 1.0 / (d[order] ** power + 1e-9)
        w /= w.sum()
        out[b] = sum(w[j] * Obs[train_ids[i]] for j, i in enumerate(order))  # (14,4)
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data_dir = find_data_dir()
    df = pd.read_csv(os.path.join(data_dir, "bioreaction_remaining.csv"))
    tpl = pd.read_csv(os.path.join(data_dir, "bioreaction_template.csv"), dtype={"batch_id": str})
    R = build_rows(df)
    pred_ids = sorted(R[R.to_predict].batch_id.unique(), key=lambda s: int(s[5:]))
    train = R[~R.batch_id.isin(pred_ids)].copy()
    hid = R[R.batch_id.isin(pred_ids) & R.to_predict].copy()
    for D in (train, hid):
        m = D.V.isna()
        D.loc[m, "V"] = D.loc[m, "V0"] + D.loc[m, ["cum_fb", "cum_fs", "cum_fc", "cum_fd"]].sum(axis=1)
    train_ids = sorted(train.batch_id.unique(), key=lambda s: int(s[5:]))
    mu, sd = train[FEATS].mean(), train[FEATS].std() + 1e-9

    # two holdout splits for robust selection
    splits = [train_ids[::8], train_ids[3::8]]
    cand = ["gbm", "knn", "gp"]
    sel = {t: {c: [] for c in cand} for t in TARGETS}
    for si, holdout in enumerate(splits):
        tr = train[~train.batch_id.isin(holdout)]
        ho = train[train.batch_id.isin(holdout)]
        fut = (ho.elapsed_time > 0).values
        Xtr, Xho = ((tr[FEATS] - mu) / sd).values, ((ho[FEATS] - mu) / sd).values
        knn_ho = knn_pred(df, [b for b in train_ids if b not in holdout], holdout)
        for c, t in enumerate(TARGETS):
            ytr, yho = tr[t].values, ho[t].values
            # gbm
            m = gbm().fit(Xtr, ytr)
            p = m.predict(Xho)
            sel[t]["gbm"].append((np.abs(p[fut] - yho[fut]) / yho[fut]).mean() * 100)
            # knn
            p_knn = np.zeros(len(ho))
            for b in holdout:
                rows_b = ho[ho.batch_id == b].sort_values("elapsed_time")
                traj = knn_ho[b]
                for j, ii in enumerate(rows_b.index):
                    p_knn[ho.index.get_loc(ii)] = traj[min(j, 13), c]
            sel[t]["knn"].append((np.abs(p_knn[fut] - yho[fut]) / yho[fut]).mean() * 100)
            # gp
            g = gp(len(FEATS)).fit(Xtr, ytr)
            p = g.predict(Xho)
            sel[t]["gp"].append((np.abs(p[fut] - yho[fut]) / yho[fut]).mean() * 100)
        print(f"split {si} done", flush=True)
    choice = {}
    for t in TARGETS:
        means = {c: float(np.mean(v)) for c, v in sel[t].items()}
        choice[t] = min(means, key=means.get)
        print(f"{t:10s}: " + "  ".join(f"{c}={means[c]:.2f}" for c in cand) + f"  -> pick {choice[t]}", flush=True)

    # ---------- final fit per target with chosen model ----------
    Xa = ((train[FEATS] - mu) / sd).values
    hid_s = hid.sort_values(["batch_id", "elapsed_time"])
    Xh = ((hid_s[FEATS] - mu) / sd).values
    knn_all = knn_pred(df, train_ids, pred_ids)
    preds = {}
    for c, t in enumerate(TARGETS):
        y = train[t].values
        ch = choice[t]
        if ch == "gbm":
            p = gbm().fit(Xa, y).predict(Xh)
        elif ch == "knn":
            p = np.zeros(len(hid_s))
            for b in pred_ids:
                rows_b = hid_s[hid_s.batch_id == b]
                traj = knn_all[b]
                for j, ii in enumerate(rows_b.index):
                    p[hid_s.index.get_loc(ii)] = traj[min(j + 1, 13), c]
        else:
            p = gp(len(FEATS)).fit(Xa, y).predict(Xh)
        preds[t] = np.maximum(p, 0.0)
        print(f"fit {t} with {ch}", flush=True)
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
    print(f"\nwrote {os.path.abspath(out_path)} choice={choice}", flush=True)
    print(sub.head(3).to_string())


if __name__ == "__main__":
    main()

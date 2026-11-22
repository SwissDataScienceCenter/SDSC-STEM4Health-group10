"""Best reproducible model (leaderboard mean_mape 8.657 for Team 10).

Per-target prediction for the 12 hidden batches:
  - vcd, tcd, byproduct: distance-weighted kNN over RICH batch features
    (initial state + cumulative feed_b/feed_substrate per day + daily feed_b
    histogram), k=3, weights 1/d, trajectories aligned by observation index.
  - substrate: 50/50 blend of the same kNN and a HistGradientBoosting model on
    per-row features (t, initial state, cumulative feeds, V(t)).

Run: python3 src/best_model.py   (writes ../submission.csv)
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from simple_ml import build_rows, find_data_dir, FEATS, TARGETS, FEEDS


def rich_feats(df, b):
    g = df[df.batch_id == b].sort_values("elapsed_time")
    o = g[g.vcd.notna()].reset_index(drop=True)
    f = g[g[FEEDS].notna().any(axis=1)]
    ft, fa = f.elapsed_time.values, f[FEEDS].fillna(0.0).values
    days = np.arange(0.5, 13.5, 1.0)
    cumfb = np.array([fa[ft <= d, 1].sum() for d in days])
    cumfs = np.array([fa[ft <= d, 0].sum() for d in days])
    hist = np.histogram(ft[fa[:, 1] > 0], bins=np.arange(0, 14, 1),
                        weights=fa[fa[:, 1] > 0, 1])[0]
    return np.r_[[o.vcd.iloc[0], o.tcd.iloc[0] - o.vcd.iloc[0], o.substrate.iloc[0],
                  o.byproduct.iloc[0], o.volume.iloc[0]], cumfb, cumfs, hist]


def main():
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

    # kNN on rich features, k=3, 1/d weights
    F = np.stack([rich_feats(df, b) for b in train_ids])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fn = (F - mu) / sd
    Obs = {b: df[(df.batch_id == b) & df.vcd.notna()].sort_values("elapsed_time")[TARGETS].values
           for b in train_ids}
    knn = {}
    for b in pred_ids:
        d = ((Fn - (rich_feats(df, b) - mu) / sd) ** 2).sum(1)
        order = np.argsort(d)[:3]
        w = 1.0 / (d[order] + 1e-9)
        w /= w.sum()
        knn[b] = sum(w[j] * Obs[train_ids[i]] for j, i in enumerate(order))  # (14,4)

    # GBM for substrate (per-row features)
    m = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.07, max_leaf_nodes=31,
                                      min_samples_leaf=6, l2_regularization=1.0, random_state=0)
    m.fit(train[FEATS], train["substrate"])

    rows = []
    for b in pred_ids:
        tt = tpl[tpl.batch_id == b].sort_values("elapsed_time").elapsed_time.values
        Xb = hid[hid.batch_id == b].sort_values("elapsed_time")[FEATS]
        s_gbm = m.predict(Xb)
        for j, t in enumerate(tt):
            row = {"batch_id": b, "elapsed_time": t}
            for c, tc in enumerate(TARGETS):
                v = knn[b][min(j + 1, 13), c]
                if tc == "substrate":
                    v = 0.5 * v + 0.5 * s_gbm[j]
                row[tc] = max(v, 0.0)
            rows.append(row)
    sub = pd.DataFrame(rows)
    tpl["_k"] = tpl.batch_id + "|" + tpl.elapsed_time.round(6).astype(str)
    sub["_k"] = sub.batch_id + "|" + sub.elapsed_time.round(6).astype(str)
    out = tpl.drop(columns=TARGETS).merge(sub, on=["_k", "batch_id", "elapsed_time"], how="left")
    sub = out[["batch_id", "elapsed_time"] + TARGETS]
    assert sub.notna().all().all() and np.isfinite(sub[TARGETS].values).all()
    out_path = os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"wrote {os.path.abspath(out_path)} shape={sub.shape}")


if __name__ == "__main__":
    main()

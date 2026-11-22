"""Simple ML approach: per-row supervised model.

Each row = (batch, time t). Features: initial state, full feed schedule
descriptors, cumulative feeds up to t, logged volume at t, and t itself.
One HistGradientBoosting model per target. Honest validation = holdout batches.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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


def build_rows(df):
    """One row per (batch, observed-or-to-predict time) with features."""
    rows = []
    for b, g in df.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        obs0 = g[g.vcd.notna()]
        y0row = obs0.iloc[0]
        f = g[g[FEEDS].notna().any(axis=1)]
        f_t = f.elapsed_time.values.astype(float)
        f_a = f[FEEDS].fillna(0.0).values.astype(float)
        tot = f_a.sum(0) if len(f_a) else np.zeros(4)
        # rows to predict: to_predict rows if any, else observed rows
        if g.to_predict.any():
            out = g[g.to_predict]
        else:
            out = g[g.vcd.notna()]
        for _, r in out.iterrows():
            t = float(r.elapsed_time)
            m = f_t <= t + 1e-9
            cum = f_a[m].sum(0) if m.any() else np.zeros(4)
            rows.append(dict(
                batch_id=b, elapsed_time=t, to_predict=bool(r.to_predict),
                vcd0=float(y0row.vcd), xd0=float(y0row.tcd - y0row.vcd),
                S0=float(y0row.substrate), P0=float(y0row.byproduct), V0=float(y0row.volume),
                n_f=len(f_t), tot_fb=float(tot[1]), tot_fs=float(tot[0]),
                tot_fc=float(tot[2]), tot_fd=float(tot[3]),
                first_fb=float(f_t[f_a[:, 1] > 0][0]) if (f_a[:, 1] > 0).any() else 0.0,
                cum_fb=float(cum[1]), cum_fs=float(cum[0]),
                cum_fc=float(cum[2]), cum_fd=float(cum[3]), n_f_so_far=int(m.sum()),
                V=float(r.volume) if pd.notna(r.volume) else np.nan,
                vcd=r.vcd if pd.notna(r.vcd) else np.nan,
                tcd=r.tcd if pd.notna(r.tcd) else np.nan,
                substrate=r.substrate if pd.notna(r.substrate) else np.nan,
                byproduct=r.byproduct if pd.notna(r.byproduct) else np.nan,
            ))
    return pd.DataFrame(rows)


FEATS = ["elapsed_time", "vcd0", "xd0", "S0", "P0", "V0", "n_f", "tot_fb", "tot_fs",
         "tot_fc", "tot_fd", "first_fb", "cum_fb", "cum_fs", "cum_fc", "cum_fd",
         "n_f_so_far", "V"]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--holdout-every", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data_dir = find_data_dir()
    df = pd.read_csv(os.path.join(data_dir, "bioreaction_remaining.csv"))
    tpl = pd.read_csv(os.path.join(data_dir, "bioreaction_template.csv"), dtype={"batch_id": str})
    R = build_rows(df)
    train = R[~R.to_predict].copy()
    hid = R[R.to_predict].copy()
    train_ids = sorted(train.batch_id.unique(), key=lambda s: int(s[5:]))
    print(f"train rows: {len(train)} ({len(train_ids)} batches), hidden rows: {len(hid)}", flush=True)

    # fill NaN V with V0 + cum feeds (fallback)
    for D in (train, hid):
        m = D.V.isna()
        D.loc[m, "V"] = D.loc[m, "V0"] + D.loc[m, ["cum_fb", "cum_fs", "cum_fc", "cum_fd"]].sum(axis=1)

    holdout = train_ids[::args.holdout_every]
    tr = train[~train.batch_id.isin(holdout)]
    ho = train[train.batch_id.isin(holdout)]

    models = {}
    for t in TARGETS:
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08,
                                          max_leaf_nodes=31, min_samples_leaf=8,
                                          l2_regularization=1.0, random_state=0)
        m.fit(tr[FEATS], tr[t])
        models[t] = m
        pred = m.predict(ho[FEATS])
        # future points only (exclude t0 row)
        ho_f = ho[ho.elapsed_time > 0]
        pred_f = m.predict(ho_f[FEATS])
        mape = (np.abs(pred_f - ho_f[t]) / ho_f[t]).mean() * 100
        print(f"holdout MAPE {t:10s}: {mape:6.2f}%", flush=True)

    # full fit for prediction
    for t in TARGETS:
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08,
                                          max_leaf_nodes=31, min_samples_leaf=8,
                                          l2_regularization=1.0, random_state=0)
        m.fit(train[FEATS], train[t])
        models[t] = m

    hid = hid.copy()
    for t in TARGETS:
        hid[t + "_pred"] = np.maximum(models[t].predict(hid[FEATS]), 0.0)

    # merge on rounded time key (CSVs store different float precision)
    tpl["_k"] = tpl.batch_id + "|" + tpl.elapsed_time.round(6).astype(str)
    hid["_k"] = hid.batch_id + "|" + hid.elapsed_time.round(6).astype(str)
    out = tpl.merge(hid[["_k"] + [t + "_pred" for t in TARGETS]], on="_k", how="left")
    assert out[[t + "_pred" for t in TARGETS]].notna().all().all()
    sub = out[["batch_id", "elapsed_time"]].copy()
    for t in TARGETS:
        sub[t] = out[t + "_pred"]
    assert sub.notna().all().all() and np.isfinite(sub[TARGETS].values).all()
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nwrote {os.path.abspath(out_path)} shape={sub.shape}", flush=True)
    print(sub.head(3).to_string())


if __name__ == "__main__":
    main()

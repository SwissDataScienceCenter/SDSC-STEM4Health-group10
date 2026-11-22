"""Cascade simple-ML model with physics-inspired features.

Stage 1: vcd ~ (t, initial state, feed schedule, cumulative feeds, V)
Stage 2+: substrate, byproduct, tcd ~ same + predicted vcd and its cumulative
integral (biomass integral drives substrate uptake / byproduct production).

Train-time features for the cascade use out-of-fold predictions (no leakage).
Honest validation = holdout batches, MAPE on the 13 future points.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold

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
    rows = []
    for b, g in df.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        obs0 = g[g.vcd.notna()]
        y0row = obs0.iloc[0]
        f = g[g[FEEDS].notna().any(axis=1)]
        f_t = f.elapsed_time.values.astype(float)
        f_a = f[FEEDS].fillna(0.0).values.astype(float)
        tot = f_a.sum(0) if len(f_a) else np.zeros(4)
        if g.to_predict.any():
            out = pd.concat([obs0.iloc[[0]], g[g.to_predict]])  # include t0 row for integrals
        else:
            out = g[g.vcd.notna()]
        for _, r in out.iterrows():
            t = float(r.elapsed_time)
            m = f_t <= t + 1e-9
            cum = f_a[m].sum(0) if m.any() else np.zeros(4)
            # recency windows
            def cwin(dt):
                mm = (f_t > t - dt) & (f_t <= t + 1e-9)
                return f_a[mm].sum(0) if mm.any() else np.zeros(4)
            c05, c1, c2 = cwin(0.5), cwin(1.0), cwin(2.0)
            last_fb = float(f_t[f_a[:, 1] > 0][-1]) if (f_a[:, 1] > 0).any() else 0.0
            rows.append(dict(
                batch_id=b, elapsed_time=t, to_predict=bool(r.to_predict),
                vcd0=float(y0row.vcd), xd0=float(y0row.tcd - y0row.vcd),
                S0=float(y0row.substrate), P0=float(y0row.byproduct), V0=float(y0row.volume),
                n_f=len(f_t), tot_fb=float(tot[1]), tot_fs=float(tot[0]),
                tot_fc=float(tot[2]), tot_fd=float(tot[3]),
                first_fb=float(f_t[f_a[:, 1] > 0][0]) if (f_a[:, 1] > 0).any() else 0.0,
                cum_fb=float(cum[1]), cum_fs=float(cum[0]),
                cum_fc=float(cum[2]), cum_fd=float(cum[3]), n_f_so_far=int(m.sum()),
                cum_fb_1d=float(c1[1]), cum_fb_2d=float(c2[1]), cum_fs_1d=float(c1[0]),
                t_since_fb=float(t - last_fb),
                V=float(r.volume) if pd.notna(r.volume) else np.nan,
                vcd=r.vcd if pd.notna(r.vcd) else np.nan,
                tcd=r.tcd if pd.notna(r.tcd) else np.nan,
                substrate=r.substrate if pd.notna(r.substrate) else np.nan,
                byproduct=r.byproduct if pd.notna(r.byproduct) else np.nan,
            ))
    return pd.DataFrame(rows)


FEATS = ["elapsed_time", "vcd0", "xd0", "S0", "P0", "V0", "n_f", "tot_fb", "tot_fs",
         "tot_fc", "tot_fd", "first_fb", "cum_fb", "cum_fs", "cum_fc", "cum_fd",
         "n_f_so_far", "V", "cum_fb_1d", "cum_fb_2d", "cum_fs_1d", "t_since_fb"]


def add_cascade_features(R):
    """Add predicted-vcd and cumulative biomass integral per row (within batch, by time)."""
    R = R.copy()
    R["vcd_hat"] = np.nan
    R["int_vcd"] = np.nan
    for b, g in R.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        v = g["vcd_hat"].values
        t = g["elapsed_time"].values
        iv = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))]) if len(t) > 1 else np.zeros(1)
        R.loc[g.index, "int_vcd"] = iv
    return R


def gbm():
    return HistGradientBoostingRegressor(max_iter=800, learning_rate=0.07,
                                         max_leaf_nodes=31, min_samples_leaf=6,
                                         l2_regularization=1.0, random_state=0)


def usable_feats(D, feats):
    return [f for f in feats if D[f].nunique() > 1]


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
    pred_ids = sorted(R[R.to_predict].batch_id.unique(), key=lambda s: int(s[5:]))
    train = R[~R.batch_id.isin(pred_ids)].copy()
    hid = R[R.batch_id.isin(pred_ids)].copy()   # includes t0 row (used only for integrals)
    for D in (train, hid):
        m = D.V.isna()
        D.loc[m, "V"] = D.loc[m, "V0"] + D.loc[m, ["cum_fb", "cum_fs", "cum_fc", "cum_fd"]].sum(axis=1)
    train_ids = sorted(train.batch_id.unique(), key=lambda s: int(s[5:]))
    print(f"train rows: {len(train)} ({len(train_ids)} batches), hidden rows: {len(hid)}", flush=True)

    holdout = train_ids[::args.holdout_every]
    tr = train[~train.batch_id.isin(holdout)].copy()
    ho = train[train.batch_id.isin(holdout)].copy()

    # ---- stage 1: vcd, with OOF predictions for cascade features on train ----
    F1 = usable_feats(tr, FEATS)
    m_vcd = gbm().fit(tr[F1], tr["vcd"])
    oof_vcd = cross_val_predict(gbm(), train[F1], train["vcd"], cv=KFold(5, shuffle=True, random_state=0))
    train["vcd_hat"] = oof_vcd
    tr["vcd_hat"] = train.loc[tr.index, "vcd_hat"]
    ho["vcd_hat"] = m_vcd.predict(ho[F1])
    train = add_cascade_features(train)
    tr = train.loc[tr.index]
    ho = add_cascade_features(pd.concat([ho, ]))  # ho alone: integral over its own rows

    F2 = usable_feats(tr, FEATS + ["vcd_hat", "int_vcd"])
    # ---- validation of each stage on holdout ----
    results = {}
    stage_feats = {"substrate": F2, "byproduct": F2, "tcd": usable_feats(tr, FEATS + ["vcd_hat"])}
    models = {"vcd": m_vcd}
    for t in ["substrate", "byproduct", "tcd"]:
        m = gbm().fit(tr[stage_feats[t]], tr[t])
        models[t] = m
        pred = m.predict(ho[stage_feats[t]])
        ho_f = ho[ho.elapsed_time > 0]
        mape = (np.abs(m.predict(ho_f[stage_feats[t]]) - ho_f[t]) / ho_f[t]).mean() * 100
        results[t] = mape
        print(f"holdout MAPE {t:10s}: {mape:6.2f}%", flush=True)
    ho_f = ho[ho.elapsed_time > 0]
    mape_vcd = (np.abs(m_vcd.predict(ho_f[F1]) - ho_f["vcd"]) / ho_f["vcd"]).mean() * 100
    results["vcd"] = mape_vcd
    print(f"holdout MAPE vcd       : {mape_vcd:6.2f}%", flush=True)
    print(f"holdout mean MAPE: {np.mean(list(results.values())):.2f}%", flush=True)

    # ---- full fit on all train, predict hidden ----
    m_vcd_full = gbm().fit(train[F1], train["vcd"])
    train["vcd_hat"] = m_vcd_full.predict(train[F1])
    train = add_cascade_features(train)
    hid["vcd_hat"] = m_vcd_full.predict(hid[F1])
    hid = add_cascade_features(hid)
    preds = {"vcd": m_vcd_full.predict(hid[F1])}
    for t in ["substrate", "byproduct", "tcd"]:
        m = gbm().fit(train[stage_feats[t]], train[t])
        preds[t] = np.maximum(m.predict(hid[stage_feats[t]]), 0.0)
    preds["vcd"] = np.maximum(preds["vcd"], 0.0)

    tpl["_k"] = tpl.batch_id + "|" + tpl.elapsed_time.round(6).astype(str)
    hid["_k"] = hid.batch_id + "|" + hid.elapsed_time.round(6).astype(str)
    out = tpl.drop(columns=TARGETS).copy()
    for t in TARGETS:
        hh = hid[["_k"]].copy()
        hh[t] = preds[t]
        out = out.merge(hh, on="_k", how="left")
        assert out[t].notna().all(), t
    sub = out[["batch_id", "elapsed_time"] + TARGETS].copy()
    assert sub.notna().all().all() and np.isfinite(sub[TARGETS].values).all()
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nwrote {os.path.abspath(out_path)} shape={sub.shape}", flush=True)
    print(sub.head(3).to_string())


if __name__ == "__main__":
    main()

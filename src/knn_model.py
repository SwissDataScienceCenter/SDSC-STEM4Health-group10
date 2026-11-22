"""Weighted kNN trajectory matching — the simple model that works best here.

Underlying relation found in the data: batches are a homogeneous cohort —
each batch's 14-point trajectory is largely determined by its initial state
(vcd0, xd0, S0, P0, V0) and its feed schedule (counts/totals per feed type).
So for a hidden batch, find the k most similar training batches in that
feature space and take their distance-weighted average trajectory, aligned by
observation index (all batches are sampled on ~daily grids).

Honest validation: holdout batches, MAPE over the 13 future points.
"""
import os
import numpy as np
import pandas as pd

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


def batch_feats(df, b):
    g = df[df.batch_id == b].sort_values("elapsed_time")
    o = g[g.vcd.notna()].reset_index(drop=True)
    f = g[g[FEEDS].notna().any(axis=1)]
    fa = f[FEEDS].fillna(0.0).values
    return np.array([
        o.vcd.iloc[0], o.tcd.iloc[0] - o.vcd.iloc[0], o.substrate.iloc[0],
        o.byproduct.iloc[0], o.volume.iloc[0],
        len(f), fa[:, 0].sum(), fa[:, 1].sum(), fa[:, 2].sum(), fa[:, 3].sum(),
    ], float)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--power", type=float, default=2.0)
    p.add_argument("--holdout-every", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data_dir = find_data_dir()
    df = pd.read_csv(os.path.join(data_dir, "bioreaction_remaining.csv"))
    tpl = pd.read_csv(os.path.join(data_dir, "bioreaction_template.csv"), dtype={"batch_id": str})
    train_ids = sorted([b for b, g in df.groupby("batch_id") if g.vcd.notna().sum() >= 14],
                       key=lambda s: int(s[5:]))
    pred_ids = sorted([b for b, g in df.groupby("batch_id") if g.to_predict.any()],
                      key=lambda s: int(s[5:]))

    F = np.stack([batch_feats(df, b) for b in train_ids])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fn = (F - mu) / sd
    Obs = {b: df[(df.batch_id == b) & df.vcd.notna()].sort_values("elapsed_time")[TARGETS].values
           for b in train_ids}

    def predict_traj(b, tr_ids, Fn_tr, k, power):
        d = ((Fn_tr - (batch_feats(df, b) - mu) / sd) ** 2).sum(1)
        order = np.argsort(d)[:k]
        w = 1.0 / (d[order] ** power + 1e-9)
        w /= w.sum()
        return sum(w[j] * Obs[tr_ids[i]] for j, i in enumerate(order))  # (14,4)

    # ---------- honest validation ----------
    ho = train_ids[::args.holdout_every]
    tr = [b for b in train_ids if b not in ho]
    Fn_tr = np.stack([Fn[train_ids.index(b)] for b in tr])
    res = {t: [] for t in TARGETS}
    for b in ho:
        traj = predict_traj(b, tr, Fn_tr, args.k, args.power)
        true = Obs[b]
        for c, t in enumerate(TARGETS):
            res[t].append(np.abs(traj[1:, c] - true[1:, c]) / true[1:, c])
    print("holdout MAPE%: " + "  ".join(f"{t} {100*np.mean(res[t]):.2f}" for t in TARGETS)
          + f"  mean {100*np.mean([np.mean(res[t]) for t in TARGETS]):.2f}", flush=True)

    # ---------- final predictions for hidden batches ----------
    rows = []
    for b in pred_ids:
        traj = predict_traj(b, train_ids, Fn, args.k, args.power)
        tt = tpl[tpl.batch_id == b].sort_values("elapsed_time").elapsed_time.values
        for j, t in enumerate(tt):
            rows.append({"batch_id": b, "elapsed_time": t,
                         **{TARGETS[c]: max(traj[min(j + 1, 13), c], 0.0) for c in range(4)}})
    sub = pd.DataFrame(rows)
    tpl["_k"] = tpl.batch_id + "|" + tpl.elapsed_time.round(6).astype(str)
    sub["_k"] = sub.batch_id + "|" + sub.elapsed_time.round(6).astype(str)
    out = tpl.drop(columns=TARGETS).merge(sub, on=["_k", "batch_id", "elapsed_time"], how="left")
    sub = out[["batch_id", "elapsed_time"] + TARGETS]
    assert sub.notna().all().all() and np.isfinite(sub[TARGETS].values).all()
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"wrote {os.path.abspath(out_path)} shape={sub.shape}")


if __name__ == "__main__":
    main()

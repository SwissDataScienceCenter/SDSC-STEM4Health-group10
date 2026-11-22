"""Experiment 2: per-target geometric ensemble of LOBO members, weights tuned by NESTED CV.

Members are read from their saved LOBO out-of-fold predictions (results/overnight/<name>_oof.csv). For each target
and each held-out batch b, the weights (simplex grid, step `step`) minimising MAPE on the other 48 batches are
chosen, and applied to b's member predictions -> an honest estimate of the tuned ensemble. The final weights
(used for the submission) are the ones that minimise MAPE on all 49 batches.

Usage: python src/experiments/ensemble.py NAME member1 member2 [member3 ...] [--step 0.1]
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as X  # noqa: E402


def simplex(n, step):
    k = int(round(1 / step))
    return np.array([c for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]) / k


def load_members(names):
    return {n: X._sorted(pd.read_csv(X.OUT / f"{n}_oof.csv")) for n in names}


def nested_weights(members: dict, step=0.1):
    names = list(members)
    t = X.truth()
    W = simplex(len(names), step)                                   # (n_w, n_members)
    out = X._sorted(members[names[0]]).copy()
    final, chosen = {}, {}
    for v in X.TARGETS:
        L = np.stack([np.log(members[n][v].to_numpy()) for n in names], 1)   # (n_pts, n_members)
        P = np.exp(L @ W.T)                                                  # (n_pts, n_w)
        ape = np.abs(P - t[v].to_numpy()[:, None]) / t[v].to_numpy()[:, None]
        bid = t.batch_id.to_numpy()
        tot, n_tot = ape.sum(0), len(ape)
        picks = []
        for b in np.unique(bid):
            r = bid == b
            k = np.argmin((tot - ape[r].sum(0)) / (n_tot - r.sum()))   # best weights on the other 48 batches
            out.loc[r, v] = P[r, k]
            picks.append(tuple(W[k]))
        final[v] = dict(zip(names, W[np.argmin(tot)]))
        chosen[v] = pd.Series(picks).value_counts().head(3).to_dict()
    return out, final, chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("members", nargs="+")
    ap.add_argument("--step", type=float, default=0.1)
    a = ap.parse_args()
    mem = load_members(a.members)
    for n, m in mem.items():
        print(f"  member {n}: " + "  ".join(f"{k}={v:.3f}" for k, v in X.score(m).items()))
    oof, final, chosen = nested_weights(mem, a.step)
    X.report(a.name, oof)
    print("final weights (fit on all 49):", {v: {k: round(w, 2) for k, w in d.items()} for v, d in final.items()})
    print("nested picks (top 3 per target, count over 49 folds):", chosen)
    pd.DataFrame(final).to_csv(X.OUT / f"{a.name}_weights.csv")

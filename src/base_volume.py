"""Hidden base volume: a read-out of lactate (byproduct) production from the reactor volume.

Biology: cells make lactic acid during the overflow phase (days ~2-5); the pH controller compensates with base,
which is NOT in the feed records but adds liquid to the tank. Hence, per sampling interval,
    hidden volume = (volume at this sample - volume at the previous sample) - all recorded feeds in between
(negative on average because of sampling withdrawals; only differences between batches matter).
    base_production = hidden volume summed over the intervals ending at samples 3, 4 and 5
Train batches: corr(base_production, peak byproduct) = 0.72. The volume is given at every sampling time of the
test batches too, so this is a legitimate input (no measurement of the test batch is used).
Idea from the team note "Lactate clue: hidden base volume".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data as D

BASE_INTERVALS = (3, 4, 5)


def hidden_volumes(raw: pd.DataFrame) -> pd.DataFrame:
    """batch_id x j (j = 1..13): unexplained volume change over the interval ending at sample j."""
    samp = D.measurements(raw, include_to_predict=True)
    f = D.feeds_long(raw)
    rows = []
    for b, g in samp.groupby("batch_id"):
        g = g.sort_values("elapsed_time").reset_index(drop=True)
        fb = f[f.batch_id == b]
        for i in range(1, len(g)):
            fed = fb[(fb.elapsed_time > g.elapsed_time[i - 1]) & (fb.elapsed_time <= g.elapsed_time[i])].amount.sum()
            rows.append({"batch_id": b, "j": i, "hidden": g.volume[i] - g.volume[i - 1] - fed})
    return pd.DataFrame(rows).pivot(index="batch_id", columns="j", values="hidden")


def base_production(raw: pd.DataFrame) -> pd.Series:
    return hidden_volumes(raw)[list(BASE_INTERVALS)].sum(axis=1).rename("base_production")


def add_base_features(table: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Adds base_production (static) and hidden_cum_t (hidden volume accumulated up to sample j) to a table
    with columns batch_id, j."""
    H = hidden_volumes(raw)
    t = table.copy()
    t["base_production"] = t.batch_id.map(H[list(BASE_INTERVALS)].sum(axis=1))
    C = H.cumsum(axis=1)
    t["hidden_cum_t"] = [C.loc[b, int(j)] for b, j in zip(t.batch_id, t.j)]
    return t


# --------------------------------------------------------------------------
# Refined versions (src/experiments/base_opt.py, choice on CV seeds 0-2, checked on 3-5)
# --------------------------------------------------------------------------
# In the late intervals (samples 6-13, no base addition) the hidden volume is withdrawal + evaporation:
# hidden ~ a + b * interval length (b < 0). Subtracting this baseline from every interval gives the base added
# per interval; summing consecutive intervals cancels the volume measurement noise (adjacent intervals are
# anti-correlated). Best windows: intervals 3-5 for the byproduct (CV 7.92 -> 7.90, 7.84 -> 7.82) and 2-5 for the
# substrate (per-time ridge 6.93 -> 6.72, 7.12 -> 6.91). Only volumes and feed times are used (inputs).
LATE_INTERVALS = list(range(6, 14))


def corrected_hidden(raw: pd.DataFrame) -> pd.DataFrame:
    H = hidden_volumes(raw)
    samp = D.measurements(raw, include_to_predict=True)
    samp["j"] = samp.groupby("batch_id").cumcount()
    dt = samp.pivot(index="batch_id", columns="j", values="elapsed_time").diff(axis=1)[list(H.columns)]
    b, a = np.polyfit(dt[LATE_INTERVALS].to_numpy().ravel(), H[LATE_INTERVALS].to_numpy().ravel(), 1)
    return H - (a + b * dt)


def base_features(raw: pd.DataFrame) -> pd.DataFrame:
    Hc = corrected_hidden(raw)
    return pd.DataFrame({"base_c345": Hc[[3, 4, 5]].sum(axis=1), "base_c2345": Hc[[2, 3, 4, 5]].sum(axis=1)})


def add_refined_base_features(table: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    t = add_base_features(table, raw)
    bf = base_features(raw)
    for c in bf.columns:
        t[c] = t.batch_id.map(bf[c])
    return t

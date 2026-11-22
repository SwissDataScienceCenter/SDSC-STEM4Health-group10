"""Feed B is dosed by a SUBSTRATE feedback controller: its dose sizes reveal the measured substrate.

Finding (49 training batches): the change of the Feed B dose from one day to the next follows the substrate
measured just before it: corr( log(dose_k / dose_k-1), log(S_k / S_k-1) ) = -0.5 to -0.85 (days 2-9), i.e.
the controller increases Feed B (which contains substrate) when the substrate drops. A log-linear controller
law  log dose_k ~ log dose_k-1 + log S_k + log S_k-1 (+ VCD, volume)  explains 55-78 % of the dose variation.
Since substrate consumption is driven by the cells, the dose sequence also carries information on the VCD.

The test batches come with their full feed profile, so these doses are legitimate inputs (no measurement of
the test batch is used). This module adds, for every row (batch, sample j), the log Feed B doses AROUND the
sample (dose after sample j-1, after sample j, after sample j+1) and their log ratios.
Bolus k (k = 1..12) follows sample k (median delay 0.05 day).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data as D

FB_LOCAL = ["lfb_prev", "lfb_next", "lfb_next2", "dlfb_prev", "dlfb_next"]


def add_feedb_features(table: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    f = D.feeds_long(raw)
    b = f[f.feed == "feed_b"].sort_values(["batch_id", "elapsed_time"]).copy()
    b["k"] = b.groupby("batch_id").cumcount() + 1
    L = np.log(b.pivot(index="batch_id", columns="k", values="amount"))            # (batch, k = 1..12)
    L = L.reindex(columns=range(0, 15)).T.bfill().ffill().T                        # pad k = 0 and k = 13, 14
    t = table.copy()
    get = lambda k: np.array([L.loc[bid, kk] for bid, kk in zip(t.batch_id, k)])
    j = t.j.to_numpy().astype(int)
    t["lfb_prev"], t["lfb_next"], t["lfb_next2"] = get(j - 1), get(j), get(j + 1)  # doses after samples j-1, j, j+1
    t["dlfb_prev"] = t.lfb_next - t.lfb_prev                                       # dose change around sample j
    t["dlfb_next"] = t.lfb_next2 - t.lfb_next
    for k in range(1, 13):                                                         # whole log-dose profile
        t[f"lfb_{k}"] = t.batch_id.map(L[k])
    return t

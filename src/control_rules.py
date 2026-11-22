"""Substrate control rule read from the feed profile (post-processing of substrate predictions).

Observed on the 49 training batches: between samples 5 and 10 the process controller adds an EXTRA substrate
bolus right after a sample whose substrate is below ~1.9 (7 / 49 training batches, triggered after sample
7, 8 or 9; measured substrate at the trigger 1.80-1.91, one batch 1.14). The dose is proportional to the
volume, so the substrate at the next sample is ~3.1 (2.7-3.25). Batches without an extra bolus never go
below 1.91 in that window. Hence, from the (given) feed profile of a test batch:
    * extra bolus after sample j0   ->  S(j0) ~ c_trig,  S(j0 + 1) ~ c_after
    * no extra bolus after sample j ->  S(j) >= floor   (j in the control window)
All three numbers are estimated on the training batches (median / minimum).
Cross-validation (9 seeds): substrate MAPE on the extra-bolus batches 10.2 % -> 9.0 %, others unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data as D

WINDOW = range(5, 11)        # sample indices where the extra-bolus rule is active


def trigger_samples(raw: pd.DataFrame) -> pd.Series:
    """batch_id -> index of the sample after which an extra (mid-run) substrate bolus was given."""
    m = D.measurements(raw, include_to_predict=True)
    m["j"] = m.groupby("batch_id").cumcount()
    f = D.feeds_long(raw)
    f = f[f.feed == "feed_substrate"].sort_values("elapsed_time")
    f = pd.merge_asof(f, m[["batch_id", "elapsed_time", "j"]].rename(columns={"elapsed_time": "t_s"}).sort_values("t_s"),
                      left_on="elapsed_time", right_on="t_s", by="batch_id", direction="backward")
    return f[f.j < 11].groupby("batch_id").j.min()       # regular boluses come after samples 11 and 12


class SubstrateControlRule:
    def fit(self, raw: pd.DataFrame, train_ids: list[str]):
        self.trig = trigger_samples(raw)
        m = D.measurements(raw)
        m["j"] = m.groupby("batch_id").cumcount()
        s = m[m.batch_id.isin(train_ids)].set_index(["batch_id", "j"]).substrate
        ex = [b for b in train_ids if b in self.trig.index]
        nex = [b for b in train_ids if b not in self.trig.index]
        self.floor = s.loc[nex].reset_index().query("j in @WINDOW").groupby("batch_id").substrate.min().min()
        self.c_trig = float(np.median([s.loc[(b, self.trig[b])] for b in ex]))
        self.c_after = float(np.median([s.loc[(b, self.trig[b] + 1)] for b in ex]))
        return self

    def apply(self, pred: pd.DataFrame) -> pd.DataFrame:
        """pred: batch_id, j, substrate (+ other columns). Returns a corrected copy."""
        p = pred.copy()
        jt = p.batch_id.map(self.trig)
        in_win = p.j.isin(WINDOW) & (jt.isna() | (p.j != jt))
        p.loc[in_win, "substrate"] = np.maximum(p.loc[in_win, "substrate"], self.floor)
        p.loc[p.j == jt, "substrate"] = self.c_trig
        p.loc[p.j == jt + 1, "substrate"] = self.c_after
        return p


# --------------------------------------------------------------------------
# Late substrate doses (after samples 11 and 12): two dose levels that reveal the measured substrate
# --------------------------------------------------------------------------
# Every batch gets a substrate bolus ~0.5 day after samples 11 and 12. Its size per volume takes only two
# values: 0.006 x V when the substrate measured at that sample was below ~0.96, 0.004 x V otherwise (a few
# exceptions). The dose class of a test batch is known from its feed profile, so the substrate predicted at
# samples 11 / 12 is clipped to the side of the threshold given by the dose:
#     big dose   -> S(j) <= cap   = quantile (1 - Q) of the training S(j) of big-dose samples
#     small dose -> S(j) >= floor = quantile Q of the training S(j) of small-dose samples
# Q = 0.2 chosen on CV seeds 0-2 (substrate MAPE -0.116), confirmed on fresh seeds 3-5 (-0.169).
LATE_SAMPLES = (11, 12)
BIG_DOSE_PER_V = 0.005


def late_dose_classes(raw: pd.DataFrame) -> pd.Series:
    """(batch_id, j) -> True if the substrate dose after sample j (j = 11, 12) is the big one."""
    samp = D.measurements(raw, include_to_predict=True)
    samp["j"] = samp.groupby("batch_id").cumcount()
    f = D.feeds_long(raw)
    f = f[f.feed == "feed_substrate"].sort_values("elapsed_time")
    f = pd.merge_asof(f, samp[["batch_id", "elapsed_time", "j", "volume"]].rename(columns={"elapsed_time": "t_s"})
                      .sort_values("t_s"), left_on="elapsed_time", right_on="t_s", by="batch_id", direction="backward")
    f = f[f.j.isin(LATE_SAMPLES)]
    return (f.amount / f.volume > BIG_DOSE_PER_V).set_axis(pd.MultiIndex.from_arrays([f.batch_id, f.j]))


class LateDoseRule:
    def __init__(self, q: float = 0.2):
        self.q = q

    def fit(self, raw: pd.DataFrame, train_ids: list[str]):
        self.big = late_dose_classes(raw)
        m = D.measurements(raw)
        m["j"] = m.groupby("batch_id").cumcount()
        s = m.set_index(["batch_id", "j"]).substrate
        idx = [(b, j) for b in train_ids for j in LATE_SAMPLES]
        y, c = s.loc[idx], self.big.loc[idx].to_numpy()
        self.cap, self.floor = float(y[c].quantile(1 - self.q)), float(y[~c].quantile(self.q))
        return self

    def apply(self, pred: pd.DataFrame) -> pd.DataFrame:
        """pred: batch_id, j, substrate (+ other columns). Returns a corrected copy."""
        p = pred.copy()
        for i, (b, j) in enumerate(zip(p.batch_id, p.j)):
            if (b, j) in self.big.index:
                k = p.index[i]
                p.loc[k, "substrate"] = (min(p.loc[k, "substrate"], self.cap) if self.big.loc[(b, j)]
                                         else max(p.loc[k, "substrate"], self.floor))
        return p

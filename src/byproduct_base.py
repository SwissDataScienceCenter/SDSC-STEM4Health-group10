"""Byproduct (lactate) model with the hidden base volume (team note "Lactate clue: hidden base volume").

Biology: lactate is made in a burst on days ~1-4 (overflow metabolism), then slowly consumed. Lactic acid lowers
the pH, the controller adds base, the base is not in the feed records but shows up in the tank volume:
base_production (src/base_volume.py, hidden volume summed over the intervals ending at samples 3-5) is a direct
read-out of how much lactate the batch made (corr 0.72 with peak lactate on the 49 training batches).

Model: one robust (Huber) regression per sampling index on log(lactate) - typical log value, blended 80/20 with
the typical log curve of the training batches.
CV on our folds (7 folds by batch, 6 seeds, src/experiments/families_cv.py byphuber): byproduct MAPE 7.88 %
vs 8.67 % for the teammate model used so far. Final blend: 0.9 this model + 0.1 teammate (chosen on seeds 0-2:
7.90 %, fresh seeds 3-5: 7.83 %).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import base_volume as BV
import data as D
import regression as R

SMALL_C = 0.0013          # Feed C doses below this volume are the small, control-triggered ones
FEATURES = ["log_vcd0", "log_tcd0", "log_byproduct0", "V0", "feed_substrate_tot", "feed_b_tot", "Vt", "feed_b_cum_t",
            "nCs_tot", "base_production"]
W_TYPICAL = 0.2
W_BLEND = 0.9             # weight of this model vs the teammate's byproduct in the final submission


def build_table(raw: pd.DataFrame) -> pd.DataFrame:
    tab = BV.add_base_features(R.build_table(raw), raw)
    f = D.feeds_long(raw)
    small = f[(f.feed == "feed_c") & (f.amount <= SMALL_C)].groupby("batch_id").size()
    tab["nCs_tot"] = tab.batch_id.map(small).fillna(0)
    return tab


# Refined variant (src/experiments/base_opt.py): evaporation-corrected base over intervals 3-5 (base_c345, see
# src/base_volume.py), Huber alpha 0.3, epsilon 2.0: CV 7.90 / 7.82 -> 7.88 / 7.81 (seeds 0-2 / 3-5).
FEATURES_C = [f for f in FEATURES if f != "base_production"] + ["base_c345"]
REFINED = dict(features=FEATURES_C, alpha=0.3, epsilon=2.0, max_iter=2000)


# + mid-run substrate bolus indicator (control regime from the feed profile: fast glucose consumers make more
# lactate), Huber epsilon 2.0: CV 7.90 / 7.83 -> 7.75 / 7.73 (0.9 this + 0.1 teammate, seeds 0-2 / 3-5).
FEATURES_TRIG = FEATURES + ["trig"]
TRIG = dict(features=FEATURES_TRIG, alpha=0.3, epsilon=2.0, max_iter=2000)


def add_trigger(tab: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    import control_rules as C
    t = tab.copy()
    t["trig"] = t.batch_id.isin(C.trigger_samples(raw).index).astype(float)
    return t


def predict(tab: pd.DataFrame, train_ids, ids, features=FEATURES, alpha=0.3, epsilon=1.35, max_iter=1000) -> pd.DataFrame:
    """Fit on train_ids, predict ids. Returns batch_id, j, elapsed_time, byproduct.
    For the refined variant: build the table with base_volume.add_refined_base_features and pass **REFINED."""
    tr = tab[tab.batch_id.isin(train_ids)]
    te = tab[tab.batch_id.isin(ids)][["batch_id", "j", "elapsed_time", *features]].copy()
    te["byproduct"] = np.nan
    for j, g in te.groupby("j"):
        gt = tr[tr.j == j]
        typical = np.log(gt.byproduct).mean()
        model = make_pipeline(StandardScaler(), HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=max_iter))
        model.fit(gt[features], np.log(gt.byproduct) - typical)
        te.loc[g.index, "byproduct"] = np.exp((1 - W_TYPICAL) * (model.predict(g[features]) + typical) + W_TYPICAL * typical)
    return te[["batch_id", "j", "elapsed_time", "byproduct"]].reset_index(drop=True)

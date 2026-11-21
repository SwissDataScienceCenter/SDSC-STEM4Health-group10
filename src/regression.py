"""Data-driven model: one regression per (target variable, sampling index), on log targets.

Each batch has 14 samples (index j = 0..13, roughly one per day). For every j >= 1 and every target we fit
    log y_j  ~  features(batch, t_j)
on the training batches, where the features only use what is known for a test batch:
    * the t = 0 measurement (vcd, tcd, substrate, byproduct, volume),
    * the feed profile (every bolus: time and volume),
    * the reactor volume at the sampling time (given on the to_predict rows),
    * optionally, the prediction of a mechanistic model simulated from the same inputs ("hybrid" model).
Log targets make the squared loss close to a relative-error loss, i.e. aligned with the leaderboard MAPE.

The default learner is ridge regression with standardised features and the penalty chosen by internal
cross-validation (RidgeCV), refitted separately for each (variable, j).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data as D

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
KEYS = ["batch_id", "j", "elapsed_time"]
ALPHAS = np.logspace(-2, 4, 25)


# --------------------------------------------------------------------------
# Feature table
# --------------------------------------------------------------------------
def _batch_static(m0: pd.Series, fb: pd.DataFrame) -> dict:
    """Features that do not depend on the sampling time."""
    x = {f"log_{v}0": np.log(m0[v]) for v in TARGETS}
    x["V0"] = m0["volume"]
    for name in D.FEEDS:
        x[f"{name}_tot"] = fb.loc[fb.feed == name, "amount"].sum()
    B = fb[fb.feed == "feed_b"].sort_values("elapsed_time")
    for i, a in enumerate(B["amount"].to_numpy()):              # each Feed B bolus (12 per batch)
        x[f"fb_{i}"] = a
    s = fb[fb.feed == "feed_substrate"].sort_values("elapsed_time")
    x["fs_t_first"] = s["elapsed_time"].iloc[0]                  # when substrate ran low (feed trigger)
    x["fs_first_amt"] = s["amount"].iloc[0]
    c = fb[fb.feed == "feed_c"].sort_values("elapsed_time")
    x["fc_t_first"] = c["elapsed_time"].iloc[0]
    return x


def _batch_dynamic(fb: pd.DataFrame, t: float) -> dict:
    """Features that depend on the sampling time t: feeds delivered up to t."""
    x = {}
    for name in D.FEEDS:
        h = fb[(fb.feed == name) & (fb.elapsed_time <= t)]
        x[f"{name}_cum_t"] = h["amount"].sum()
        x[f"{name}_n_t"] = len(h)
    return x


def build_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per batch and sampling time t > 0 (train AND test batches), with features and targets.
    Targets are NaN on the to_predict rows."""
    m = D.measurements(raw, include_to_predict=True)
    m["j"] = m.groupby("batch_id").cumcount()
    f = D.feeds_long(raw)
    rows = []
    for b, g in m.groupby("batch_id", sort=False):
        fb = f[f.batch_id == b]
        static = _batch_static(g.iloc[0], fb)
        for _, r in g[g.j > 0].iterrows():
            rows.append({"batch_id": b, "j": r.j, "elapsed_time": r.elapsed_time,
                         **{v: r[v] for v in TARGETS}, "dcd": r["dcd"],
                         **static, "t": r.elapsed_time, "Vt": r.volume, **_batch_dynamic(fb, r.elapsed_time)})
    return pd.DataFrame(rows)


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in KEYS + TARGETS + ["dcd"]]


# Selected feature sets (see results/ridge_blend/README.md for the selection procedure):
#  * dropped for all targets: Feed C and Feed D (negligible volume, unknown composition) and the 12 individual
#    Feed B bolus sizes, which only added noise in cross-validation;
#  * then per-target backward elimination on CV seeds 0-2, checked on fresh seeds 3-5.
REDUCED = ["log_vcd0", "log_tcd0", "log_substrate0", "log_byproduct0", "V0",
           "feed_substrate_tot", "feed_b_tot", "fs_t_first", "fs_first_amt", "t", "Vt",
           "feed_substrate_cum_t", "feed_substrate_n_t", "feed_b_cum_t", "feed_b_n_t"]
FEATURES_BY_TARGET = {
    "vcd": REDUCED,
    "tcd": REDUCED,
    "substrate": [c for c in REDUCED if c not in ("log_byproduct0", "feed_substrate_n_t", "log_vcd0")],
    "byproduct": [c for c in REDUCED if c not in ("log_substrate0", "fs_t_first", "fs_first_amt", "t",
                                                  "feed_substrate_n_t", "feed_substrate_cum_t")],
}


def add_mechanistic(table: pd.DataFrame, pred: pd.DataFrame, prefix: str = "mech") -> pd.DataFrame:
    """Hybrid model: attach a mechanistic model's predictions (log) as extra features."""
    p = pred.assign(_t=pred.elapsed_time.round(6))[["batch_id", "_t", *TARGETS]]
    p = p.rename(columns={v: f"{prefix}_log_{v}" for v in TARGETS})
    for v in TARGETS:
        p[f"{prefix}_log_{v}"] = np.log(np.maximum(p[f"{prefix}_log_{v}"], 1e-3))
    out = table.assign(_t=table.elapsed_time.round(6)).merge(p, on=["batch_id", "_t"], how="left")
    return out.drop(columns="_t")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


class PerTimeRegression:
    """One regressor per (target, sampling index j), fitted on log(target).

    `features`: None (all feature columns), a list (same features for every target) or a dict target -> list.
    """

    def __init__(self, features: list[str] | dict | None = None, make_model=ridge):
        self.features, self.make_model = features, make_model

    def _cols(self, v: str) -> list[str]:
        return self.features[v] if isinstance(self.features, dict) else self.features

    def fit(self, table: pd.DataFrame, train_ids: list[str]):
        tr = table[table.batch_id.isin(train_ids)]
        self.features = self.features or feature_columns(table)
        self.models = {}
        for v in TARGETS:
            for j, g in tr.groupby("j"):
                self.models[v, j] = self.make_model().fit(g[self._cols(v)], np.log(g[v]))
        return self

    def predict(self, table: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
        te = table[table.batch_id.isin(ids)]
        out = te[KEYS].copy()
        for v in TARGETS:
            out[v] = np.nan
            for j, g in te.groupby("j"):
                out.loc[g.index, v] = np.exp(self.models[v, j].predict(g[self._cols(v)]))
        out["dcd"] = out["tcd"] - out["vcd"]
        return out.reset_index(drop=True)

    def coefficients(self, v: str) -> pd.DataFrame:
        """Standardised ridge coefficients per sampling index (rows = features, columns = j):
        effect on log(target) of a one-standard-deviation change of each input."""
        return pd.DataFrame({j: self.models[v, j][-1].coef_ for (vv, j) in self.models if vv == v},
                            index=self._cols(v))

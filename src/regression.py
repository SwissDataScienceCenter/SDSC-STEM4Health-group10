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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, RidgeCV
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
    x["nC_early"] = int((c["elapsed_time"] < 5).sum())          # on-demand Feed C boluses during growth (< day 5)
    # on-demand Feed C boluses at the end of the run (day >= 11), log(1 + n) to limit extrapolation for the
    # rare batches with many late boluses. Hypothesis: Feed C is a pH-control additive whose demand follows
    # byproduct (lactate-like) consumption; empirically, more late Feed C <-> lower byproduct.
    x["nC_late_log"] = np.log1p(int((c["elapsed_time"] >= 11).sum()))
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


# Selected feature sets (selection procedure, all scored by 7-fold CV by batch, MAPE):
#  * dropped for all targets: Feed C and Feed D (negligible volume, unknown composition) and the 12 individual
#    Feed B bolus sizes, which only added noise in cross-validation;
#  * then per-target backward elimination on CV seeds 0-2, checked on fresh seeds 3-5;
#  * Feed C counts for byproduct (nC_early: boluses before day 5; nC_late_log: from day 11) improved the
#    byproduct in CV (nC_late_log: +0.49 +- 0.21 points per batch) but made it WORSE on the leaderboard twice
#    (8.78 -> 9.20 and 9.09): not robust on independent batches -> not used (features kept in the table).
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
# Number of eigencurves for the FPCA projection of the predicted trajectories (None = no projection).
# Substrate is not projected: its trajectory jumps at feed boluses and is not well summarised by 2 modes.
# Byproduct: projection is neutral in CV, left out (simpler).
N_EIGENCURVES = {"vcd": 2, "tcd": 2, "substrate": None, "byproduct": None}


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


class ScaledRidgeCV(BaseEstimator, RegressorMixin):
    """RidgeCV (efficient leave-one-out choice of the penalty), then refit with penalty x k.
    k < 1 = weaker shrinkage than the leave-one-out optimum (chosen per target by grouped CV)."""

    def __init__(self, k: float = 1.0):
        self.k = k

    def fit(self, X, y):
        self.alpha_ = RidgeCV(alphas=ALPHAS).fit(X, y).alpha_
        self.model_ = Ridge(alpha=self.alpha_ * self.k).fit(X, y)
        self.coef_ = self.model_.coef_
        return self

    def predict(self, X):
        return self.model_.predict(X)


def scaled_ridge(k: float):
    return lambda: make_pipeline(StandardScaler(), ScaledRidgeCV(k))


class PerTimeRegression:
    """One regressor per (target, sampling index j), fitted on log(target).

    `features`: None (all feature columns), a list (same features for every target) or a dict target -> list.
    `n_eigencurves`: optional dict target -> K. The 13 per-time predictions of a batch are independent, so its
        predicted trajectory can be jagged. With K set, the predicted log trajectory is projected onto the mean
        training curve + its K main modes of variation (functional PCA of the measured training trajectories):
        lower variance, smoother curves.
    """

    def __init__(self, features: list[str] | dict | None = None, make_model=ridge, n_eigencurves: dict | None = None):
        # make_model: a model factory, or a dict target -> factory
        self.features, self.make_model, self.n_eigencurves = features, make_model, n_eigencurves or {}

    def _factory(self, v: str):
        return self.make_model[v] if isinstance(self.make_model, dict) else self.make_model

    def _cols(self, v: str) -> list[str]:
        return self.features[v] if isinstance(self.features, dict) else self.features

    def fit(self, table: pd.DataFrame, train_ids: list[str]):
        tr = table[table.batch_id.isin(train_ids)]
        self.features = self.features or feature_columns(table)
        self.models, self.fpca = {}, {}
        for v in TARGETS:
            for j, g in tr.groupby("j"):
                self.models[v, j] = self._factory(v)().fit(g[self._cols(v)], np.log(g[v]))
            K = self.n_eigencurves.get(v)
            if K:   # modes of variation of the measured log trajectories (rows = batches, columns = j)
                Z = np.log(tr.pivot(index="batch_id", columns="j", values=v))
                mu = Z.mean().to_numpy()
                _, _, Vt = np.linalg.svd(Z.to_numpy() - mu, full_matrices=False)
                self.fpca[v] = (list(Z.columns), mu, Vt[:K].T)      # (13,), (13, K)
        return self

    def predict(self, table: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
        te = table[table.batch_id.isin(ids)]
        out = te[KEYS].copy()
        for v in TARGETS:
            out[v] = np.nan
            for j, g in te.groupby("j"):
                out.loc[g.index, v] = self.models[v, j].predict(g[self._cols(v)])     # log scale
            if v in self.fpca:
                js, mu, P = self.fpca[v]
                W = out.pivot(index="batch_id", columns="j", values=v)[js]           # full 13-point trajectories
                Wp = mu + (W.to_numpy() - mu) @ P @ P.T
                proj = pd.DataFrame(Wp, index=W.index, columns=js).stack()
                out[v] = proj.loc[list(zip(out.batch_id, out.j))].to_numpy()
            out[v] = np.exp(out[v])
        out["dcd"] = out["tcd"] - out["vcd"]
        return out.reset_index(drop=True)

    def coefficients(self, v: str) -> pd.DataFrame:
        """Standardised ridge coefficients per sampling index (rows = features, columns = j):
        effect on log(target) of a one-standard-deviation change of each input."""
        return pd.DataFrame({j: self.models[v, j][-1].coef_ for (vv, j) in self.models if vv == v},
                            index=self._cols(v))


# --------------------------------------------------------------------------
# Analogue ("nearest batches") baseline
# --------------------------------------------------------------------------
ANALOG_FEATURES = ["log_vcd0", "log_tcd0", "log_substrate0", "log_byproduct0", "V0",
                   "feed_b_tot", "feed_substrate_tot", "fs_t_first"]


class NearestBatches:
    """Predicts a batch as the geometric mean trajectory of the k training batches closest to it in
    standardised initial state / feed totals (sample index by sample index)."""

    def __init__(self, k: int = 8, features: list[str] = ANALOG_FEATURES):
        self.k, self.features = k, features

    def fit(self, table: pd.DataFrame, train_ids: list[str]):
        S = table[table.j == 1].set_index("batch_id")[self.features]
        self.mu, self.sd = S.loc[train_ids].mean(), S.loc[train_ids].std()
        self.Z_train = (S.loc[train_ids] - self.mu) / self.sd
        self.logY = {v: np.log(table[table.batch_id.isin(train_ids)].pivot(index="batch_id", columns="j", values=v))
                     for v in TARGETS}
        return self

    def predict(self, table: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
        te = table[table.batch_id.isin(ids)]
        S = (te[te.j == 1].set_index("batch_id")[self.features] - self.mu) / self.sd
        out = te[KEYS].copy()
        for b in S.index:
            nn = np.sqrt(((self.Z_train - S.loc[b]) ** 2).sum(axis=1)).nsmallest(self.k).index
            rows = out.batch_id == b
            for v in TARGETS:
                out.loc[rows, v] = np.exp(self.logY[v].loc[nn].mean()).loc[out.loc[rows, "j"]].to_numpy()
        out["dcd"] = out["tcd"] - out["vcd"]
        return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Final ensemble configuration (chosen on CV seeds 0-2, confirmed on seeds 3-8)
# --------------------------------------------------------------------------
# Penalty multiplier of the per-time ridge (1 = leave-one-out optimum).
ALPHA_SCALE = {"vcd": 1.0, "tcd": 1.0, "substrate": 0.25, "byproduct": 0.25}
# Geometric blend weights: ridge / VCD0-scaled mean-curve baseline / nearest-batches baseline.
BLEND = {"vcd": (0.70, 0.15, 0.15), "tcd": (0.70, 0.15, 0.15),
         "substrate": (0.80, 0.20, 0.0), "byproduct": (0.80, 0.20, 0.0)}
# Weight of the hybrid ODE (model v3 with batch-specific parameters predicted from the batch features,
# src/ode_paramnet.py) in a final geometric blend with the result above. Cell densities only: the v3 structure
# cannot reproduce substrate / byproduct (chosen on CV seed 0, confirmed on seeds 1-5).
ODE_WEIGHT = {"vcd": 0.2, "tcd": 0.2, "substrate": 0.0, "byproduct": 0.0}

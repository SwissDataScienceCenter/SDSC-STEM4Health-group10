"""Team 10 - "Decode the Bioreactor" (SDSC STEM4Health hackathon): final solution in one file.

Rebuilds our best leaderboard submission (mean MAPE 7.300: vcd 7.164, tcd 7.044, substrate 7.349,
byproduct 7.644; file submissions/blend_base_kalman3_full.csv) from the raw data only.

Usage (from the repo root):
    python solution.py [--data-dir DIR] [--out submissions/solution.csv] [--n-jobs 8]
Default data dir: ../input-data/Project 1 - DTB (or env var DTB_DATA_DIR). Runtime ~10-20 min (the teammate's ODE
is refitted on every cross-validation fold to estimate the uncertainty used by the substrate update).

How each target is predicted (all inputs are known for a test batch: t = 0 measurement, full feed profile,
reactor volume at every sampling time):

  vcd, tcd   geometric blend  0.25 team + 0.55 SVR + 0.20 trees,  tcd >= vcd
             team  = geometric blend of model A (ours) and model B (teammate), weight on B: vcd 0.6, tcd 0.5
             SVR   = pooled-time RBF SVR (two feature sets), trees = random forest, LightGBM, extra-trees;
                     "pooled time": one model on all (batch, day) rows, target = log y - mean log curve
  substrate  prior = 0.5 model A + 0.5 model B (geometric), then a Gaussian (Kalman-type) update that uses the
             process controllers as indirect sensors:
               * Feed B is dosed after every sample by a controller that replaces the substrate consumed the
                 day before: S_k = a + b S_(k-1) + c d_k/V + e d_(k-1)/V (+ substrate-feed dose), S_0 measured
               * late substrate doses after samples 11 / 12 come in two sizes: big if S < ~0.96 (cap / floor)
               * mid-run extra bolus when S < ~1.9 between samples 5 and 10 (S at the trigger ~1.85, ~3.1 the
                 sample after, S >= 1.91 otherwise in that window)
             prior covariance = out-of-fold relative errors of the prior (7 folds x 6 seeds); then late dose rule
  byproduct  0.9 per-day Huber regression on log(byproduct) with the hidden base volume + 0.1 model B
             hidden volume = volume change - recorded feeds; in sampling intervals 3-5 it is the base added by the
             pH controller to neutralise the lactate produced (corr 0.72 with peak byproduct)

  model A (ours)    per-day ridge on log targets (projected on 2 eigencurves for cells) blended with the
                    VCD0-scaled mean curve and the 8 nearest training batches; cells blended 20 % with a hybrid
                    ODE (mechanistic model v3 with batch-specific parameters predicted by ridge from the batch
                    features); mid-run substrate bolus rule
  model B (teammate) per-day regression (ridge, cells pooled over +-3 neighbouring days; Huber for byproduct),
                    20 % typical curve, then geometric blend with a fitted ODE (growth, death, inhibitor,
                    overflow, feeds as boluses): vcd/tcd 20 %, substrate 10 % (40 % after an early top-up)

All weights / settings were chosen by grouped cross-validation (7 folds by batch, seeds 0-2, checked on 3-5).
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DTB_DATA_DIR", REPO.parent / "input-data" / "Project 1 - DTB"))
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
J = list(range(1, 14))
N_SEEDS, N_FOLDS = 6, 7


# =====================================================================================================
# 1. Data
# =====================================================================================================
def load_raw(data_dir) -> pd.DataFrame:
    """Sparse event log sorted by batch and time, with row_type in {measurement, feed, to_predict}."""
    df = pd.read_csv(Path(data_dir) / "bioreaction_remaining.csv", dtype={"batch_id": str})
    df["to_predict"] = df["to_predict"].astype(bool)
    df["batch_num"] = df["batch_id"].str.extract(r"(\d+)").astype(int)
    is_meas = df[TARGETS].notna().any(axis=1)
    is_feed = df[FEEDS].notna().any(axis=1)
    df["row_type"] = np.select([df["to_predict"], is_meas, is_feed], ["to_predict", "measurement", "feed"], default="unknown")
    return df.sort_values(["batch_num", "elapsed_time"]).reset_index(drop=True)


def _natural(ids) -> list[str]:
    return sorted(set(ids), key=lambda s: int("".join(c for c in s if c.isdigit())))


def test_batches(df):
    return _natural(df.loc[df["to_predict"], "batch_id"])


def train_batches(df):
    return _natural(set(df["batch_id"]) - set(test_batches(df)))


def measurements(df, include_to_predict=False) -> pd.DataFrame:
    """One row per sampling time (+ dcd = tcd - vcd)."""
    types = ["measurement", "to_predict"] if include_to_predict else ["measurement"]
    m = df.loc[df["row_type"].isin(types), ["batch_id", "batch_num", "elapsed_time", *TARGETS, "volume", "to_predict"]].copy()
    m["dcd"] = m["tcd"] - m["vcd"]
    m["viability"] = m["vcd"] / m["tcd"]
    return m.reset_index(drop=True)


def feeds_long(df) -> pd.DataFrame:
    """One row per feed bolus: batch_id, elapsed_time, feed, amount (volume added)."""
    f = df.loc[df["row_type"] == "feed", ["batch_id", "batch_num", "elapsed_time", *FEEDS]]
    f = f.melt(id_vars=["batch_id", "batch_num", "elapsed_time"], value_vars=FEEDS, var_name="feed",
               value_name="amount").dropna(subset=["amount"])
    return f.sort_values(["batch_num", "elapsed_time"]).reset_index(drop=True)


def make_folds(batch_ids, n_folds=7, seed=0):
    """Random partition of the batches into n_folds groups (grouped cross-validation by batch)."""
    rng = np.random.default_rng(seed)
    ids = np.array(batch_ids)[rng.permutation(len(batch_ids))]
    return [sorted(f.tolist(), key=lambda s: int(s[5:])) for f in np.array_split(ids, n_folds)]


def sample_index(raw) -> pd.DataFrame:
    s = measurements(raw, include_to_predict=True)
    s["j"] = s.groupby("batch_id").cumcount()
    return s


# =====================================================================================================
# 2. Feature table: one row per (batch, sampling index j >= 1)
# =====================================================================================================
def _batch_static(m0, fb) -> dict:
    x = {f"log_{v}0": np.log(m0[v]) for v in TARGETS}
    x["V0"] = m0["volume"]
    for name in FEEDS:
        x[f"{name}_tot"] = fb.loc[fb.feed == name, "amount"].sum()
    B = fb[fb.feed == "feed_b"].sort_values("elapsed_time")
    for i, a in enumerate(B["amount"].to_numpy()):
        x[f"fb_{i}"] = a
    s = fb[fb.feed == "feed_substrate"].sort_values("elapsed_time")
    x["fs_t_first"] = s["elapsed_time"].iloc[0]
    x["fs_first_amt"] = s["amount"].iloc[0]
    c = fb[fb.feed == "feed_c"].sort_values("elapsed_time")
    x["fc_t_first"] = c["elapsed_time"].iloc[0]
    x["nC_early"] = int((c["elapsed_time"] < 5).sum())
    x["nC_late_log"] = np.log1p(int((c["elapsed_time"] >= 11).sum()))
    return x


def _batch_dynamic(fb, t) -> dict:
    x = {}
    for name in FEEDS:
        h = fb[(fb.feed == name) & (fb.elapsed_time <= t)]
        x[f"{name}_cum_t"] = h["amount"].sum()
        x[f"{name}_n_t"] = len(h)
    return x


def build_table(raw) -> pd.DataFrame:
    m = measurements(raw, include_to_predict=True)
    m["j"] = m.groupby("batch_id").cumcount()
    f = feeds_long(raw)
    rows = []
    for b, g in m.groupby("batch_id", sort=False):
        fb = f[f.batch_id == b]
        static = _batch_static(g.iloc[0], fb)
        for _, r in g[g.j > 0].iterrows():
            rows.append({"batch_id": b, "j": r.j, "elapsed_time": r.elapsed_time, **{v: r[v] for v in TARGETS},
                         "dcd": r["dcd"], **static, "t": r.elapsed_time, "Vt": r.volume, **_batch_dynamic(fb, r.elapsed_time)})
    return pd.DataFrame(rows)


KEYS = ["batch_id", "j", "elapsed_time"]


def feature_columns(table):
    return [c for c in table.columns if c not in KEYS + TARGETS + ["dcd"]]


REDUCED = ["log_vcd0", "log_tcd0", "log_substrate0", "log_byproduct0", "V0", "feed_substrate_tot", "feed_b_tot",
           "fs_t_first", "fs_first_amt", "t", "Vt", "feed_substrate_cum_t", "feed_substrate_n_t", "feed_b_cum_t", "feed_b_n_t"]
FEATURES_BY_TARGET = {
    "vcd": REDUCED, "tcd": REDUCED,
    "substrate": [c for c in REDUCED if c not in ("log_byproduct0", "feed_substrate_n_t", "log_vcd0")],
    "byproduct": [c for c in REDUCED if c not in ("log_substrate0", "fs_t_first", "fs_first_amt", "t",
                                                  "feed_substrate_n_t", "feed_substrate_cum_t")],
}


# =====================================================================================================
# 3. Model A (ours): per-time ridge + eigencurves + baselines + hybrid ODE + mid-run substrate rule
# =====================================================================================================
ALPHAS = np.logspace(-2, 4, 25)
N_EIGENCURVES = {"vcd": 2, "tcd": 2, "substrate": None, "byproduct": None}
ALPHA_SCALE = {"vcd": 1.0, "tcd": 1.0, "substrate": 0.25, "byproduct": 0.25}
BLEND_A = {"vcd": (0.70, 0.15, 0.15), "tcd": (0.70, 0.15, 0.15), "substrate": (0.80, 0.20, 0.0), "byproduct": (0.80, 0.20, 0.0)}
ODE_WEIGHT_A = {"vcd": 0.2, "tcd": 0.2, "substrate": 0.0, "byproduct": 0.0}
ANALOG_FEATURES = ["log_vcd0", "log_tcd0", "log_substrate0", "log_byproduct0", "V0", "feed_b_tot", "feed_substrate_tot", "fs_t_first"]


class ScaledRidgeCV(BaseEstimator, RegressorMixin):
    """RidgeCV (leave-one-out choice of the penalty), refitted with penalty x k."""

    def __init__(self, k: float = 1.0):
        self.k = k

    def fit(self, X, y):
        self.alpha_ = RidgeCV(alphas=ALPHAS).fit(X, y).alpha_
        self.model_ = Ridge(alpha=self.alpha_ * self.k).fit(X, y)
        self.coef_ = self.model_.coef_
        return self

    def predict(self, X):
        return self.model_.predict(X)


def scaled_ridge(k):
    return lambda: make_pipeline(StandardScaler(), ScaledRidgeCV(k))


class PerTimeRegression:
    """One ridge per (target, sampling index j) on log(target); optional projection of the predicted log
    trajectory on the mean training curve + its K main modes of variation (functional PCA)."""

    def __init__(self, features, make_model, n_eigencurves):
        self.features, self.make_model, self.n_eigencurves = features, make_model, n_eigencurves

    def fit(self, table, train_ids):
        tr = table[table.batch_id.isin(train_ids)]
        self.models, self.fpca = {}, {}
        for v in TARGETS:
            for j, g in tr.groupby("j"):
                self.models[v, j] = self.make_model[v]().fit(g[self.features[v]], np.log(g[v]))
            K = self.n_eigencurves.get(v)
            if K:
                Z = np.log(tr.pivot(index="batch_id", columns="j", values=v))
                mu = Z.mean().to_numpy()
                _, _, Vt = np.linalg.svd(Z.to_numpy() - mu, full_matrices=False)
                self.fpca[v] = (list(Z.columns), mu, Vt[:K].T)
        return self

    def predict(self, table, ids):
        te = table[table.batch_id.isin(ids)]
        out = te[KEYS].copy()
        for v in TARGETS:
            out[v] = np.nan
            for j, g in te.groupby("j"):
                out.loc[g.index, v] = self.models[v, j].predict(g[self.features[v]])
            if v in self.fpca:
                js, mu, P = self.fpca[v]
                W = out.pivot(index="batch_id", columns="j", values=v)[js]
                Wp = mu + (W.to_numpy() - mu) @ P @ P.T
                proj = pd.DataFrame(Wp, index=W.index, columns=js).stack()
                out[v] = proj.loc[list(zip(out.batch_id, out.j))].to_numpy()
            out[v] = np.exp(out[v])
        out["dcd"] = out["tcd"] - out["vcd"]
        return out.reset_index(drop=True)


class NearestBatches:
    """Geometric mean trajectory of the k training batches closest in standardised initial state / feed totals."""

    def __init__(self, k=8, features=ANALOG_FEATURES):
        self.k, self.features = k, features

    def fit(self, table, train_ids):
        S = table[table.j == 1].set_index("batch_id")[self.features]
        self.mu, self.sd = S.loc[train_ids].mean(), S.loc[train_ids].std()
        self.Z_train = (S.loc[train_ids] - self.mu) / self.sd
        self.logY = {v: np.log(table[table.batch_id.isin(train_ids)].pivot(index="batch_id", columns="j", values=v)) for v in TARGETS}
        return self

    def predict(self, table, ids):
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


class MeanTrajectory:
    """Average training curve per culture day, cell densities rescaled by the batch's initial VCD."""
    VARS = ["vcd", "dcd", "tcd", "substrate", "byproduct"]

    def fit(self, raw, train_ids):
        m = measurements(raw[raw.batch_id.isin(train_ids)])
        m["day"] = m["elapsed_time"].round().astype(int)
        self.curve = m.groupby("day")[["elapsed_time", *self.VARS]].mean()
        self.vcd0 = m.loc[m.elapsed_time == 0, "vcd"].mean()
        return self

    def predict(self, raw, batch_ids):
        m = measurements(raw[raw.batch_id.isin(batch_ids)], include_to_predict=True)
        x0 = m[m.elapsed_time == 0].set_index("batch_id")["vcd"]
        out = m.loc[m.elapsed_time > 0, ["batch_id", "elapsed_time"]].copy()
        for v in self.VARS:
            out[v] = np.interp(out["elapsed_time"], self.curve["elapsed_time"], self.curve[v])
            if v in ("vcd", "dcd", "tcd"):
                out[v] *= out["batch_id"].map(x0).to_numpy() / self.vcd0
        return out.reset_index(drop=True)


# ---- hybrid ODE (mechanistic model v3) --------------------------------------------------------------
# States [Xv, Xd, S, P, N, D]: viable / dead cells, substrate, byproduct, hidden nutrient (normalised to 1 at t = 0,
# brought by Feed B), byproduct damage (history variable, not diluted). Growth = Monod(S) x Monod(N); death = basal +
# nutrient depletion + byproduct damage; lysis; uptake = growth + maintenance + overflow (Hill in S, shut down by P);
# byproduct = overflow - consumption. Feeds are boluses: V+ = V + dV, C+ = (C V + C_feed dV) / V+; the volume is
# reset to the measured value at every sample. Fixed-step RK4 (dt = 0.01 d).
EPS_S = 0.05
STATE_NAMES = ["vcd", "dcd", "substrate", "byproduct"]
# shared v3 parameters, fitted once on the 49 training batches (global least-squares fit of the v3 model)
V3_SHARED = {"mu_max": 11.163179137584006, "Ks": 1.177622765474397, "K_N": 9.999999999999877, "Yxn": 45.63757623457124,
             "C_Nb": 0.0001000000061893375, "kd0": 1.0000003858148416e-06, "kd_N": 1.9999510541005938,
             "K_dN": 0.0002766707955168232, "kd_D": 0.04580845323314155, "k_r": 9.983053825024863,
             "k_lys": 0.30826027845800513, "Yxs": 9999.999999950609, "m": 0.06193105271914369, "q_ovf": 4.999999999999998,
             "K_ovf": 2.7063557158998854, "n_ovf": 19.99999999999999, "K_iP": 0.089625189523255, "Yps": 1.086215889812011,
             "qPc": 1.9999999999999998, "K_Pc": 27.335546978773724, "C_sub": 1.000000000000004, "C_b": 90.156428838729}
VARY = ["mu_max", "Yxn", "kd0", "m", "q_ovf", "qPc"]       # parameters made batch-specific (param-net)
ODE_OBS = ["vcd", "tcd", "substrate", "byproduct"]


def v3_rates(x, p):
    Xv, Xd, S, P, N, Dm = np.moveaxis(x, -1, 0)
    Sp, Pp, Np, Dp = np.maximum(S, 0.0), np.maximum(P, 0.0), np.maximum(N, 0.0), np.maximum(Dm, 0.0)
    fS = Sp / (Sp + EPS_S)
    mu = p["mu_max"] * Sp / (p["Ks"] + Sp) * Np / (p["K_N"] + Np)
    kd = p["kd0"] + p["kd_N"] * p["K_dN"] / (p["K_dN"] + Np) + p["kd_D"] * Dp
    Sn = Sp ** p["n_ovf"]
    q_ovf = p["q_ovf"] * Sn / (p["K_ovf"] ** p["n_ovf"] + Sn) * p["K_iP"] / (p["K_iP"] + Pp)
    qS = mu / p["Yxs"] + (p["m"] + q_ovf) * fS
    qP = p["Yps"] * q_ovf * fS - p["qPc"] * Pp / (p["K_Pc"] + Pp)
    return np.stack([(mu - kd) * Xv, kd * Xv - p["k_lys"] * Xd, -qS * Xv, qP * Xv, -mu / p["Yxn"] * Xv, Pp - p["k_r"] * Dp], axis=-1)


def v3_feed_conc(p):
    out = {f: np.zeros(6) for f in FEEDS}
    out["feed_substrate"][2] = p["C_sub"]
    out["feed_b"][2] = p["C_b"]
    out["feed_b"][4] = p["C_Nb"]
    return out


@dataclass
class BatchSet:
    ids: list
    dt: float
    n_steps: int
    x0: np.ndarray
    v0: np.ndarray
    vol_reset: np.ndarray
    feed_vol: dict
    obs: pd.DataFrame


def v3_prepare(raw, batch_ids, dt=0.01) -> BatchSet:
    sub = raw[raw["batch_id"].isin(batch_ids)]
    meas = measurements(sub, include_to_predict=True)
    feeds = feeds_long(sub)
    idx = {b: i for i, b in enumerate(batch_ids)}
    n_steps = int(np.ceil(meas["elapsed_time"].max() / dt)) + 2
    B = len(batch_ids)
    m0 = meas[meas["elapsed_time"] == 0].set_index("batch_id").loc[batch_ids]
    x0, v0 = m0[STATE_NAMES].to_numpy(float), m0["volume"].to_numpy(float)
    step_of = lambda t: np.maximum(np.ceil(np.asarray(t) / dt).astype(int) - 1, 0)
    vol_reset = np.full((n_steps, B), np.nan)
    later = meas[meas["elapsed_time"] > 0]
    vol_reset[step_of(later["elapsed_time"]), later["batch_id"].map(idx)] = later["volume"]
    feed_vol = {}
    for name in FEEDS:
        arr = np.zeros((n_steps, B))
        f = feeds[feeds["feed"] == name]
        np.add.at(arr, (step_of(f["elapsed_time"]), f["batch_id"].map(idx).to_numpy()), f["amount"].to_numpy())
        feed_vol[name] = arr
    obs = later[["batch_id", "elapsed_time", "vcd", "dcd", "tcd", "substrate", "byproduct"]].copy()
    obs["b"] = obs["batch_id"].map(idx)
    obs["k"] = np.floor(obs["elapsed_time"] / dt).astype(int)
    obs["w"] = obs["elapsed_time"] / dt - obs["k"]
    return BatchSet(list(batch_ids), dt, n_steps, x0, v0, vol_reset, feed_vol, obs.reset_index(drop=True))


def v3_simulate(p, bs: BatchSet):
    dt = bs.dt
    x = np.hstack([bs.x0, np.broadcast_to(np.array([1.0, 0.0]), (bs.x0.shape[0], 2))])
    V = bs.v0.copy()
    conc = v3_feed_conc(p)
    post = np.empty((bs.n_steps + 1, *x.shape)); pre = np.empty_like(post)
    post[0] = pre[0] = x
    for k in range(bs.n_steps):
        k1 = v3_rates(x, p); k2 = v3_rates(x + 0.5 * dt * k1, p)
        k3 = v3_rates(x + 0.5 * dt * k2, p); k4 = v3_rates(x + dt * k3, p)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        pre[k + 1] = x
        r = bs.vol_reset[k]
        V = np.where(np.isnan(r), V, r)
        dVs = {f: bs.feed_vol[f][k] for f in FEEDS}
        dV = sum(dVs.values())
        if dV.any():
            Vn = V + dV
            added = sum(dVs[f][:, None] * conc[f][None, :] for f in FEEDS)
            x_new = (x * V[:, None] + added) / Vn[:, None]
            x_new[:, 5] = x[:, 5]                                  # damage state is not diluted
            x, V = x_new, Vn
        post[k + 1] = x
    o = bs.obs
    w = o["w"].to_numpy()[:, None]
    val = (1 - w) * post[o["k"], o["b"], :4] + w * pre[o["k"] + 1, o["b"], :4]
    out = o[["batch_id", "elapsed_time"]].copy()
    out[STATE_NAMES] = val
    out["tcd"] = out["vcd"] + out["dcd"]
    return out


def v3_fit_per_batch(raw, ids, shared, prior_sd=0.5, res_sd=0.1, max_nfev=40):
    """Per-batch log-parameters VARY (others shared), one sparse least-squares call, Gaussian prior to shared."""
    bs = v3_prepare(raw, ids)
    B, K = len(ids), len(VARY)
    z0 = np.log(np.array([[shared[k]] * B for k in VARY]).T)
    lb, ub = z0 - np.log(20), z0 + np.log(20)
    y = np.log(bs.obs[ODE_OBS].to_numpy(float))
    w = res_sd / prior_sd

    def params(z):
        Z = np.exp(z.reshape(B, K)); p = dict(shared)
        for k, name in enumerate(VARY):
            p[name] = Z[:, k]
        return p

    def resid(z):
        with np.errstate(all="ignore"):
            pred = v3_simulate(params(z), bs)[ODE_OBS].to_numpy(float)
            r = (np.log(np.maximum(pred, 1e-6)) - y).ravel()
        r = np.where(np.isfinite(r), r, 10.0)
        return np.r_[r, w * (z - z0.ravel())]

    n_obs = len(bs.obs); S = lil_matrix((n_obs * 4 + B * K, B * K), dtype=int)
    for i, b in enumerate(bs.obs["b"].to_numpy()):
        S[4 * i:4 * i + 4, b * K:(b + 1) * K] = 1
    for q in range(B * K):
        S[n_obs * 4 + q, q] = 1
    sol = least_squares(resid, z0.ravel(), bounds=(lb.ravel(), ub.ravel()), jac_sparsity=S, x_scale="jac",
                        diff_step=1e-3, max_nfev=max_nfev, verbose=0)
    return pd.DataFrame(np.exp(sol.x.reshape(B, K)), index=ids, columns=VARY)


def hybrid_ode_predict(raw, table, train_ids, ids):
    """Per-batch ODE fit on train_ids -> ridge 'param-net' (batch features -> log parameters) -> simulate ids."""
    X = table[table.j == 1].set_index("batch_id")[ANALOG_FEATURES]
    P = v3_fit_per_batch(raw, train_ids, V3_SHARED)
    Y = np.log(P[VARY].to_numpy(float))
    nets = [make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X.loc[train_ids], Y[:, k]) for k in range(len(VARY))]
    Pids = pd.DataFrame(np.exp(np.c_[[m.predict(X.loc[ids]) for m in nets]].T), index=ids, columns=VARY)
    p = dict(V3_SHARED)
    for name in VARY:
        p[name] = Pids.loc[ids, name].to_numpy(float)
    return v3_simulate(p, v3_prepare(raw, ids))


# ---- substrate control rules ------------------------------------------------------------------------
WINDOW = range(5, 11)          # samples where the mid-run extra-bolus rule is active
LATE_SAMPLES = (11, 12)
BIG_DOSE_PER_V = 0.005


def trigger_samples(raw) -> pd.Series:
    """batch_id -> sample after which a mid-run (j < 11) substrate bolus was given."""
    m = sample_index(raw)
    f = feeds_long(raw)
    f = f[f.feed == "feed_substrate"].sort_values("elapsed_time")
    f = pd.merge_asof(f, m[["batch_id", "elapsed_time", "j"]].rename(columns={"elapsed_time": "t_s"}).sort_values("t_s"),
                      left_on="elapsed_time", right_on="t_s", by="batch_id", direction="backward")
    return f[f.j < 11].groupby("batch_id").j.min()


class SubstrateControlRule:
    """Mid-run bolus after sample j0 when S(j0) < ~1.9: S(j0) ~ c_trig, S(j0+1) ~ c_after, else S >= floor in WINDOW."""

    def fit(self, raw, train_ids):
        self.trig = trigger_samples(raw)
        m = measurements(raw)
        m["j"] = m.groupby("batch_id").cumcount()
        s = m[m.batch_id.isin(train_ids)].set_index(["batch_id", "j"]).substrate
        ex = [b for b in train_ids if b in self.trig.index]
        nex = [b for b in train_ids if b not in self.trig.index]
        self.floor = s.loc[nex].reset_index().query("j in @WINDOW").groupby("batch_id").substrate.min().min()
        self.c_trig = float(np.median([s.loc[(b, self.trig[b])] for b in ex]))
        self.c_after = float(np.median([s.loc[(b, self.trig[b] + 1)] for b in ex]))
        return self

    def apply(self, pred):
        p = pred.copy()
        jt = p.batch_id.map(self.trig)
        in_win = p.j.isin(WINDOW) & (jt.isna() | (p.j != jt))
        p.loc[in_win, "substrate"] = np.maximum(p.loc[in_win, "substrate"], self.floor)
        p.loc[p.j == jt, "substrate"] = self.c_trig
        p.loc[p.j == jt + 1, "substrate"] = self.c_after
        return p


def late_dose_classes(raw) -> pd.Series:
    """(batch_id, j) -> True if the substrate dose after sample j (11, 12) is the big one (0.006 V vs 0.004 V)."""
    samp = sample_index(raw)
    f = feeds_long(raw)
    f = f[f.feed == "feed_substrate"].sort_values("elapsed_time")
    f = pd.merge_asof(f, samp[["batch_id", "elapsed_time", "j", "volume"]].rename(columns={"elapsed_time": "t_s"}).sort_values("t_s"),
                      left_on="elapsed_time", right_on="t_s", by="batch_id", direction="backward")
    f = f[f.j.isin(LATE_SAMPLES)]
    return (f.amount / f.volume > BIG_DOSE_PER_V).set_axis(pd.MultiIndex.from_arrays([f.batch_id, f.j]))


class LateDoseRule:
    """Big late dose -> S(j) <= cap, small dose -> S(j) >= floor (training quantiles)."""

    def __init__(self, q=0.2):
        self.q = q

    def fit(self, raw, train_ids):
        self.big = late_dose_classes(raw)
        m = measurements(raw)
        m["j"] = m.groupby("batch_id").cumcount()
        s = m.set_index(["batch_id", "j"]).substrate
        idx = [(b, j) for b in train_ids for j in LATE_SAMPLES]
        y, c = s.loc[idx], self.big.loc[idx].to_numpy()
        self.cap, self.floor = float(y[c].quantile(1 - self.q)), float(y[~c].quantile(self.q))
        return self

    def apply(self, pred):
        p = pred.copy()
        for i, (b, j) in enumerate(zip(p.batch_id, p.j)):
            if (b, j) in self.big.index:
                k = p.index[i]
                p.loc[k, "substrate"] = min(p.loc[k, "substrate"], self.cap) if self.big.loc[(b, j)] else max(p.loc[k, "substrate"], self.floor)
        return p


def model_a_predict(raw, table, train_ids, ids, ode=True):
    """Model A for batches ids (sorted by batch, time): ridge / mean curve / nearest batches, hybrid ODE (cells), mid-run rule."""
    key = ["batch_id", "elapsed_time"]
    srt = lambda d: d.sort_values(key).reset_index(drop=True)
    make = {v: scaled_ridge(k) for v, k in ALPHA_SCALE.items()}
    p = srt(PerTimeRegression(FEATURES_BY_TARGET, make, N_EIGENCURVES).fit(table, train_ids).predict(table, ids))
    b = srt(MeanTrajectory().fit(raw, train_ids).predict(raw, ids))
    nb = srt(NearestBatches().fit(table, train_ids).predict(table, ids))
    for v in TARGETS:
        w = BLEND_A[v]
        p[v] = np.exp(w[0] * np.log(p[v]) + w[1] * np.log(b[v]) + w[2] * np.log(nb[v]))
    if ode:
        o = srt(hybrid_ode_predict(raw, table, train_ids, ids))
        for v in TARGETS:
            w = ODE_WEIGHT_A[v]
            if w > 0:
                p[v] = np.exp((1 - w) * np.log(p[v]) + w * np.log(o[v]))
    p = SubstrateControlRule().fit(raw, train_ids).apply(p)
    p["dcd"] = p["tcd"] - p["vcd"]
    return p


# =====================================================================================================
# 4. Model B (teammate): per-day regression + typical curve + fitted ODE
# =====================================================================================================
TM_FEEDS, TM_OBS, TM_DT, TM_T_END = FEEDS, ["vcd", "tcd", "substrate", "byproduct"], 0.02, 13.1
TM_PARAMS = {  # name: (start, lower, upper)
    "mu_max": (0.9, 0.3, 2.5), "KA": (50.0, 5, 500), "nA": (2.0, 0.5, 6), "kdA": (0.03, 1e-4, 0.5), "KdA": (200.0, 20, 2000),
    "tlag": (0.3, 0.01, 2), "kd0": (0.007, 1e-4, 0.1), "kd1": (0.03, 1e-4, 0.3), "Kd": (1.0, 0.05, 3), "klys": (0.03, 1e-4, 0.5),
    "m": (0.105, 0.01, 0.3), "Yxs": (30.0, 0.5, 2000), "q_ov": (0.65, 1e-3, 5), "Kov": (3.0, 0.5, 10), "n_ov": (5.0, 1.0, 10),
    "Kx": (3.0, 0.1, 50), "Yp": (1.0, 0.05, 3), "q_c": (0.02, 1e-5, 0.3), "Kc": (0.5, 0.01, 5), "c_B": (52.0, 20, 120),
    "c_A": (250.0, 80, 600),
}
TM_NAMES = list(TM_PARAMS)
TM_STAGES = [("cells", ["mu_max", "KA", "nA", "kdA", "KdA", "tlag", "kd0", "kd1", "Kd", "klys"], (0, 1), 40),
             ("substrate", ["m", "Yxs", "c_B", "c_A", "q_ov", "Kov", "n_ov", "Kx"], (2, 3), 25),
             ("byproduct", ["Yp", "q_c", "Kc"], (3,), 30), ("joint", TM_NAMES, (0, 1, 2, 3), 30)]
TM_REDUCED = REDUCED
TM_FEAT = {"vcd": TM_REDUCED + ["nCs_tot"], "tcd": TM_REDUCED + ["nCs_tot"],
           "substrate": [c for c in TM_REDUCED if c not in ("log_byproduct0", "feed_substrate_n_t", "log_vcd0")],
           "byproduct": [c for c in TM_REDUCED if c not in ("log_substrate0", "fs_t_first", "fs_first_amt", "t",
                                                            "feed_substrate_n_t", "feed_substrate_cum_t")] + ["nCs_tot"]}
SMALL_C = 0.0013          # Feed C doses below this volume are the small, control-triggered ones
EARLY_TOPUP_DAY = 11.0
W_MECH_SUBSTRATE_AFTER_EARLY_TOPUP = 0.4
TM_WINDOW = {"vcd": 3, "tcd": 3, "substrate": 0, "byproduct": 0}
W_BASELINE = 0.2
W_MECH = {"vcd": 0.2, "tcd": 0.2, "substrate": 0.1, "byproduct": 0.0}


def tm_prepare(d, batches):
    n_steps = int(round(TM_T_END / TM_DT)); grid = np.arange(n_steps + 1) * TM_DT; B = len(batches)
    F = {f: np.zeros((n_steps, B)) for f in TM_FEEDS}; V = np.zeros((n_steps + 1, B)); x0 = np.zeros((5, B))
    samples = []
    for j, b in enumerate(batches):
        g = d[d.batch_id == b].sort_values("elapsed_time")
        for f in TM_FEEDS:
            ev = g[g[f].notna()]
            k = np.minimum((ev.elapsed_time.values / TM_DT).astype(int), n_steps - 1)
            np.add.at(F[f][:, j], k, ev[f].values)
        vm = g[g.volume.notna()]
        cum = lambda t: sum(g.loc[g.elapsed_time <= t, f].sum() for f in TM_FEEDS)
        base = np.array([v - cum(t) for t, v in zip(vm.elapsed_time, vm.volume)])
        V[:, j] = np.interp(grid, vm.elapsed_time.values, base) + np.r_[0, np.cumsum(sum(F[f][:, j] for f in TM_FEEDS))]
        s0 = g[(g.elapsed_time == 0) & g.vcd.notna()].iloc[0]
        x0[:, j] = [s0.vcd, s0.tcd - s0.vcd, s0.substrate, s0.byproduct, 0.0]
        for _, r in g[(g.vcd.notna() | g.to_predict) & (g.elapsed_time > 0)].iterrows():
            k = min(int(np.ceil(r.elapsed_time / TM_DT - 1e-9)), n_steps)
            samples.append((j, r.elapsed_time, k, r[TM_OBS].values.astype(float), b))
    kk = np.array([max(s[2], 1) for s in samples]); jj = np.array([s[0] for s in samples])
    w = np.clip([(s[1] - (k - 1) * TM_DT) / TM_DT for s, k in zip(samples, kk)], 0, 1)
    return dict(F=F, V=V, x0=x0, samples=samples, n_steps=n_steps, kk=kk, jj=jj, w=np.array(w))


def tm_rates(x, p, t):
    """Growth inhibited by an accumulating inhibitor A (= integral of viable cells), lag, death (basal, low S,
    inhibitor), lysis, maintenance + growth + overflow uptake, overflow -> byproduct, byproduct consumption."""
    Xv, Xd, S, P, A = np.maximum(x, 0)
    An = A ** p["nA"]
    mu = p["mu_max"] * (1 - np.exp(-t / p["tlag"])) * p["KA"] ** p["nA"] / (p["KA"] ** p["nA"] + An)
    kd = p["kd0"] + p["kd1"] * p["Kd"] ** 2 / (p["Kd"] ** 2 + S ** 2) + p["kdA"] * An / (p["KdA"] ** p["nA"] + An)
    Sn = S ** p["n_ov"]
    O = p["q_ov"] * Sn / (p["Kov"] ** p["n_ov"] + Sn) * p["Kx"] / (p["Kx"] + Xv)
    qS = np.where(S > 1e-4, p["m"] + np.maximum(mu, 0) / p["Yxs"] + O, 0)
    qP = p["Yp"] * O - p["q_c"] * P / (p["Kc"] + P)
    return np.array([(mu - kd) * Xv, kd * Xv - p["klys"] * Xd, -qS * Xv, qP * Xv, Xv])


def tm_simulate(p, prep):
    F, V, n = prep["F"], prep["V"], prep["n_steps"]
    x = prep["x0"].copy(); out = np.empty((n + 1, 5, x.shape[1])); post = np.empty_like(out); out[0] = x
    for k in range(n):
        xprev, t = x, k * TM_DT
        k1 = tm_rates(x, p, t); k2 = tm_rates(x + TM_DT / 2 * k1, p, t + TM_DT / 2)
        k3 = tm_rates(x + TM_DT / 2 * k2, p, t + TM_DT / 2); k4 = tm_rates(x + TM_DT * k3, p, t + TM_DT)
        x = np.maximum(x + TM_DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0)
        post[k] = xprev; out[k + 1] = x
        add = sum(F[f][k] for f in TM_FEEDS)
        if add.any():
            x = x * (V[k + 1] - add) / V[k + 1]
            x[2] += (p["c_B"] * F["feed_b"][k] + p["c_A"] * F["feed_substrate"][k]) / V[k + 1]
    post[n] = x
    kk, jj, w = prep["kk"], prep["jj"], prep["w"]
    st = ((1 - w)[:, None] * post[kk - 1, :, jj] + w[:, None] * out[kk, :, jj])[:, :4]
    return np.c_[st[:, 0], st[:, 0] + st[:, 1], st[:, 2], st[:, 3]]


def tm_fit(prep, p, free, use, nfev):
    lo = np.log([TM_PARAMS[n][1] for n in free]); hi = np.log([TM_PARAMS[n][2] for n in free])
    th0 = np.clip(np.log([p[n] for n in free]), lo + 1e-6, hi - 1e-6)
    obs = np.array([s[3] for s in prep["samples"]])

    def res(th):
        q = dict(p, **dict(zip(free, np.exp(th))))
        r = ((tm_simulate(q, prep) - obs) / np.abs(obs))[:, list(use)]
        return np.nan_to_num(r, nan=0.0).ravel()

    sol = least_squares(res, th0, bounds=(lo, hi), loss="soft_l1", f_scale=0.15, max_nfev=nfev, diff_step=1e-3)
    return dict(p, **dict(zip(free, np.exp(sol.x))))


def tm_build_table(d):
    rows = []
    for b, g in d.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        m = g[g.vcd.notna() | g.to_predict].reset_index(drop=True)
        m0 = m.iloc[0]
        x = {f"log_{v}0": np.log(m0[v]) for v in TARGETS}
        x["V0"] = m0.volume
        for f in TM_FEEDS:
            x[f + "_tot"] = g[f].sum()
        c = g[g.feed_c.notna()]
        x["nCs_tot"] = int((c.feed_c <= SMALL_C).sum())
        s = g[g.feed_substrate.notna()]
        x["fs_t_first"] = s.elapsed_time.iloc[0]
        x["fs_first_amt"] = s.feed_substrate.iloc[0]
        for j in range(1, len(m)):
            t = m.loc[j, "elapsed_time"]; h = g[g.elapsed_time <= t]
            r = dict(batch_id=b, j=j, elapsed_time=t, t=t, Vt=m.loc[j, "volume"], **x, **{v: m.loc[j, v] for v in TARGETS})
            for f in TM_FEEDS:
                r[f + "_cum_t"] = h[f].sum(); r[f + "_n_t"] = h[f].count()
            rows.append(r)
    return pd.DataFrame(rows)


def tm_regression_predict(tr, te):
    """Per (variable, day) regression on log values (cells pooled over +-3 neighbouring days); log predictions."""
    model = {"vcd": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
             "tcd": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
             "substrate": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
             "byproduct": lambda: make_pipeline(StandardScaler(), HuberRegressor(alpha=0.3, epsilon=1.35, max_iter=1000))}
    out = te[["batch_id", "elapsed_time"]].copy()
    for v in TARGETS:
        w, cols = TM_WINDOW[v], TM_FEAT[v]
        out[v] = np.nan
        for j, g in te.groupby("j"):
            gt = tr[(tr.j - j).abs() <= w]
            mu = gt.groupby("j")[v].apply(lambda s: np.log(s).mean())
            X = gt[cols].assign(dj=gt.j - j) if w else gt[cols]
            Xt = g[cols].assign(dj=0) if w else g[cols]
            m = model[v]().fit(X, np.log(gt[v]) - gt.j.map(mu))
            out.loc[g.index, v] = m.predict(Xt) + mu.loc[j]
    return out


def tm_baseline_predict(tr, te):
    out = te[["batch_id", "elapsed_time"]].copy()
    for v in TARGETS:
        val = tr.groupby("j")[v].apply(lambda s: np.log(s).mean()).loc[te.j].values
        if v in ("vcd", "tcd"):
            val = val + (te[f"log_{v}0"].values - tr[f"log_{v}0"].mean())
        out[v] = val
    return out


def model_b_predict(d, tab, train_ids, ids):
    """Model B for batches ids: rows batch_id, elapsed_time, j + the four targets."""
    tr, te = tab[tab.batch_id.isin(train_ids)], tab[tab.batch_id.isin(ids)]
    reg, base = tm_regression_predict(tr, te), tm_baseline_predict(tr, te)
    st = te[["batch_id", "elapsed_time", "j"]].copy()
    for v in TARGETS:
        st[v + "_stat"] = (1 - W_BASELINE) * reg[v] + W_BASELINE * base[v]
    p = {k: v[0] for k, v in TM_PARAMS.items()}
    prep = tm_prepare(d, train_ids)
    for _, free, use, nf in TM_STAGES:                       # staged ODE fit on the training batches only
        p = tm_fit(prep, p, free, use, nf)
    prep_v = tm_prepare(d, ids)
    mech = pd.DataFrame(np.log(np.maximum(tm_simulate(p, prep_v), 1e-3)), columns=[o + "_mech" for o in TM_OBS])
    mech["batch_id"] = [x[4] for x in prep_v["samples"]]
    mech["_t"] = [round(x[1], 6) for x in prep_v["samples"]]
    st["_t"] = st.elapsed_time.round(6)
    st = st.merge(mech, on=["batch_id", "_t"])
    first_topup = d[d.feed_substrate.notna()].groupby("batch_id").elapsed_time.min()
    f1 = st.batch_id.map(first_topup)
    st["after_early"] = (f1 < EARLY_TOPUP_DAY) & (st.elapsed_time > f1)
    for v in TARGETS:
        w = np.where(st.after_early & (v == "substrate"), W_MECH_SUBSTRATE_AFTER_EARLY_TOPUP, W_MECH[v])
        st[v] = np.exp((1 - w) * st[v + "_stat"] + w * st[v + "_mech"])
    st["tcd"] = np.maximum(st.tcd, st.vcd)
    return st


# =====================================================================================================
# 5. Cells: pooled-time SVR and tree models
# =====================================================================================================
FEED_C = ("fc_t_first", "nC_early", "nC_late_log")
FAMILIES = {
    "lgbm": lambda: LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, verbose=-1, random_state=0),
    "extratrees": lambda: ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, n_jobs=1, random_state=0),
    "rf": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, n_jobs=1, random_state=0),
    "svr": lambda: GridSearchCV(make_pipeline(StandardScaler(), SVR(kernel="rbf")),
                                {"svr__C": [0.3, 1, 3], "svr__gamma": [0.003, 0.01, 0.03], "svr__epsilon": [0.01, 0.03]},
                                cv=GroupKFold(5), scoring="neg_mean_squared_error"),
}
KERNEL_GROUP = ["svr_sel", "svr_wide"]
TREE_GROUP = ["rf_sel", "lgbm_sel", "extratrees_sel"]
W_CELLS = {"team": 0.25, "kernel": 0.55, "trees": 0.20}
W_TEAMMATE = {"vcd": 0.6, "tcd": 0.5, "substrate": 0.5, "byproduct": 1.0}


def feature_sets(table):
    wide = [c for c in feature_columns(table) if "feed_c" not in c and c not in FEED_C]
    return {"sel": {v: sorted(set(FEATURES_BY_TARGET[v]) | {"t"}) for v in TARGETS}, "wide": {v: wide for v in TARGETS}}


def pooled_predict(table, train_ids, ids, family, fset, targets=("vcd", "tcd")):
    """One model per target on all (batch, j) training rows; target = log y - mean log curve at j."""
    cols = feature_sets(table)[fset]
    tr, te = table[table.batch_id.isin(train_ids)], table[table.batch_id.isin(ids)]
    out = te[["batch_id", "j", "elapsed_time"]].copy()
    for v in targets:
        mu = np.log(tr[v]).groupby(tr.j).mean()
        y = np.log(tr[v]) - tr.j.map(mu)
        m = FAMILIES[family]()
        if family == "svr":
            m.fit(tr[cols[v]], y, groups=tr.batch_id)
        else:
            m.fit(tr[cols[v]], y)
        out[v] = np.exp(te.j.map(mu) + m.predict(te[cols[v]]))
    return out.reset_index(drop=True)


# =====================================================================================================
# 6. Byproduct: hidden base volume + per-day Huber regression
# =====================================================================================================
BASE_INTERVALS = (3, 4, 5)
BYP_FEATURES = ["log_vcd0", "log_tcd0", "log_byproduct0", "V0", "feed_substrate_tot", "feed_b_tot", "Vt", "feed_b_cum_t",
                "nCs_tot", "base_production"]
W_TYPICAL, W_BYP = 0.2, 0.9


def hidden_volumes(raw) -> pd.DataFrame:
    """batch x j: volume change over the interval ending at sample j minus all recorded feeds in it."""
    samp = measurements(raw, include_to_predict=True)
    f = feeds_long(raw)
    rows = []
    for b, g in samp.groupby("batch_id"):
        g = g.sort_values("elapsed_time").reset_index(drop=True)
        fb = f[f.batch_id == b]
        for i in range(1, len(g)):
            fed = fb[(fb.elapsed_time > g.elapsed_time[i - 1]) & (fb.elapsed_time <= g.elapsed_time[i])].amount.sum()
            rows.append({"batch_id": b, "j": i, "hidden": g.volume[i] - g.volume[i - 1] - fed})
    return pd.DataFrame(rows).pivot(index="batch_id", columns="j", values="hidden")


def byproduct_table(raw, table):
    t = table.copy()
    H = hidden_volumes(raw)
    t["base_production"] = t.batch_id.map(H[list(BASE_INTERVALS)].sum(axis=1))
    f = feeds_long(raw)
    small = f[(f.feed == "feed_c") & (f.amount <= SMALL_C)].groupby("batch_id").size()
    t["nCs_tot"] = t.batch_id.map(small).fillna(0)
    return t


def byproduct_predict(tab, train_ids, ids):
    tr = tab[tab.batch_id.isin(train_ids)]
    te = tab[tab.batch_id.isin(ids)][["batch_id", "j", "elapsed_time", *BYP_FEATURES]].copy()
    te["byproduct"] = np.nan
    for j, g in te.groupby("j"):
        gt = tr[tr.j == j]
        typical = np.log(gt.byproduct).mean()
        m = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.3, epsilon=1.35, max_iter=1000))
        m.fit(gt[BYP_FEATURES], np.log(gt.byproduct) - typical)
        te.loc[g.index, "byproduct"] = np.exp((1 - W_TYPICAL) * (m.predict(g[BYP_FEATURES]) + typical) + W_TYPICAL * typical)
    return te[["batch_id", "j", "elapsed_time", "byproduct"]].reset_index(drop=True)


# =====================================================================================================
# 7. Substrate: Feed B controller laws + control rules as observations (Gaussian / Kalman-type update)
# =====================================================================================================
KALMAN_DAYS = [1, 2, 4, 5, 6, 7, 8, 11]
SHRINK, NOISE_SCALE, TAU_LATE, TAU_MID = 0.4, 1.5, 0.03, 0.1


class ControllerUpdate:
    """prior S ~ N(mu, Sigma), observations H S = z + eps, posterior mu + Sigma H' (H Sigma H' + R)^-1 (z - H mu)."""

    def __init__(self, raw):
        samp = sample_index(raw)
        V = samp.pivot(index="batch_id", columns="j", values="volume")
        self.S = samp.pivot(index="batch_id", columns="j", values="substrate")       # NaN on test rows
        f = feeds_long(raw)
        fb = f[f.feed == "feed_b"].sort_values(["batch_id", "elapsed_time"]).copy()
        fb["k"] = fb.groupby("batch_id").cumcount() + 1
        self.dose = fb.pivot(index="batch_id", columns="k", values="amount") / V[list(range(1, 13))].values
        self.dose[13] = 0.0
        fa = f[f.feed == "feed_substrate"].sort_values("elapsed_time")
        fa = pd.merge_asof(fa, samp[["batch_id", "elapsed_time", "j"]].rename(columns={"elapsed_time": "ts"}).sort_values("ts"),
                           left_on="elapsed_time", right_on="ts", by="batch_id", direction="forward")
        self.fa = fa.groupby(["batch_id", "j"]).amount.sum().unstack().reindex(index=V.index, columns=range(0, 14)).fillna(0.0) / V.values

    def fit(self, train_ids, rel_errors, days=KALMAN_DAYS, shrink=SHRINK):
        """Controller laws S_k ~ S_(k-1) + d_k/V + d_(k-1)/V (+ substrate feed) on the TRUE training substrate;
        rel_errors: relative out-of-fold errors of the prior (batch x j)."""
        self.laws = {}
        for k in days:
            X = pd.DataFrame({"Sp": self.S.loc[train_ids, k - 1], "d": self.dose.loc[train_ids, k]})
            if k > 1:
                X["dp"] = self.dose.loc[train_ids, k - 1]
            if (self.fa.loc[train_ids, k] > 0).sum() >= 3:
                X["fa"] = self.fa.loc[train_ids, k]
            X = X.loc[:, X.std() > 0]
            y = self.S.loc[train_ids, k]
            lr = LinearRegression().fit(X, y)
            self.laws[k] = (lr, float(np.std(y - lr.predict(X), ddof=X.shape[1] + 1)), list(X.columns))
        self.sd = rel_errors.std().to_numpy()
        self.corr = (1 - shrink) * rel_errors.corr().to_numpy() + shrink * np.eye(rel_errors.shape[1])
        return self

    def update(self, b, mu, late, mid, noise_scale=NOISE_SCALE, iters=3, tau_mid=TAU_MID):
        Sig = np.outer(self.sd * mu, self.sd * mu) * self.corr
        H, z, Rv = [], [], []
        for k, (lr, sig, cols) in self.laws.items():
            coef = dict(zip(cols, lr.coef_))
            h = np.zeros(13); h[k - 1] = 1.0
            rhs = (lr.intercept_ + coef.get("d", 0.0) * self.dose.loc[b, k] + (coef.get("dp", 0.0) * self.dose.loc[b, k - 1] if k > 1 else 0.0)
                   + coef.get("fa", 0.0) * self.fa.loc[b, k])
            if k == 1:
                rhs += coef["Sp"] * self.S.loc[b, 0]                      # S_0 is measured
            else:
                h[k - 2] = -coef["Sp"]
            H.append(h); z.append(rhs); Rv.append((noise_scale * sig) ** 2)

        def solve(H, z, Rv):
            H, z = np.array(H), np.array(z)
            return mu + Sig @ H.T @ np.linalg.inv(H @ Sig @ H.T + np.diag(Rv)) @ (z - H @ mu)

        def obs(k, val, tau):
            h = np.zeros(13); h[k - 1] = 1.0
            return h, val, tau ** 2

        j0 = mid.trig.get(b, np.nan)
        if not np.isnan(j0):                                               # mid-run bolus: S(j0), S(j0+1)
            for k, val in ((int(j0), mid.c_trig), (int(j0) + 1, mid.c_after)):
                h, v, r = obs(k, val, tau_mid); H.append(h); z.append(v); Rv.append(r)
        post = solve(H, z, Rv)
        for _ in range(iters):                                             # violated bounds -> pseudo-observations
            extra = []
            for k in LATE_SAMPLES:
                if (b, k) in late.big.index:
                    if late.big.loc[(b, k)] and post[k - 1] > late.cap: extra.append(obs(k, late.cap, TAU_LATE))
                    if not late.big.loc[(b, k)] and post[k - 1] < late.floor: extra.append(obs(k, late.floor, TAU_LATE))
            for k in WINDOW:
                if (np.isnan(j0) or k not in (j0, j0 + 1)) and post[k - 1] < mid.floor:
                    extra.append(obs(k, mid.floor, tau_mid))
            if not extra:
                break
            post = solve(H + [e[0] for e in extra], z + [e[1] for e in extra], Rv + [e[2] for e in extra])
        return np.maximum(post, 0.05)


def _oof_task(raw, d, table, tm_tab, train, s, i, fold):
    """Out-of-fold substrate predictions of models A and B for one (seed, fold)."""
    tr = [b for b in train if b not in fold]
    a = model_a_predict(raw, table, tr, fold, ode=False)               # the ODE only enters the cells of model A
    b = model_b_predict(d, tm_tab, tr, fold)
    return s, a.set_index(["batch_id", "j"]).substrate, b.set_index(["batch_id", "j"]).substrate


# =====================================================================================================
# 8. Assembly
# =====================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(REPO / "submissions" / "solution.csv"))
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    raw = load_raw(a.data_dir)
    d = pd.read_csv(Path(a.data_dir) / "bioreaction_remaining.csv")           # model B reads the raw log as is
    tpl = pd.read_csv(Path(a.data_dir) / "bioreaction_template.csv", dtype={"batch_id": str})
    train, test = train_batches(raw), test_batches(raw)
    table, tm_tab = build_table(raw), tm_build_table(d)
    key = lambda df: df.assign(_t=df["elapsed_time"].round(6))
    align = lambda p, cols: key(tpl)[["batch_id", "_t"]].merge(key(p)[["batch_id", "_t", *cols]], on=["batch_id", "_t"], how="left")
    samp = sample_index(raw)
    jt = key(tpl)[["batch_id", "_t"]].merge(key(samp)[["batch_id", "_t", "j"]], on=["batch_id", "_t"], how="left").j.astype(int).to_numpy()
    log(f"{len(train)} training batches, {len(test)} test batches")

    # --- models A and B on the test batches
    A = align(model_a_predict(raw, table, train, test), TARGETS); log("model A (ours) fitted")
    n = d.groupby("batch_id").vcd.count()
    B = align(model_b_predict(d, tm_tab, sorted(n[n > 1].index), list(dict.fromkeys(tpl.batch_id))), TARGETS); log("model B (teammate) fitted")
    team = {v: np.exp((1 - W_TEAMMATE[v]) * np.log(A[v].to_numpy()) + W_TEAMMATE[v] * np.log(B[v].to_numpy())) for v in TARGETS}
    team["tcd"] = np.maximum(team["tcd"], team["vcd"])

    # --- cells: 0.25 team + 0.55 SVR + 0.20 trees
    fam = {}
    for name in KERNEL_GROUP + TREE_GROUP:
        family, fset = name.rsplit("_", 1)
        fam[name] = align(pooled_predict(table, train, test, family, fset), ["vcd", "tcd"])
    out = tpl[["batch_id", "elapsed_time"]].copy()
    for v in ("vcd", "tcd"):
        kern = np.exp(np.mean([np.log(fam[n][v].to_numpy()) for n in KERNEL_GROUP], axis=0))
        trees = np.exp(np.mean([np.log(fam[n][v].to_numpy()) for n in TREE_GROUP], axis=0))
        out[v] = np.exp(W_CELLS["team"] * np.log(team[v]) + W_CELLS["kernel"] * np.log(kern) + W_CELLS["trees"] * np.log(trees))
    out["tcd"] = np.maximum(out.tcd, out.vcd)
    log("cells done (SVR + trees)")

    # --- byproduct: 0.9 hidden-base Huber model + 0.1 model B
    P = align(byproduct_predict(byproduct_table(raw, table), train, test), ["byproduct"]).byproduct.to_numpy()
    out["byproduct"] = np.exp(W_BYP * np.log(P) + (1 - W_BYP) * np.log(team["byproduct"]))
    log("byproduct done (hidden base volume)")

    # --- substrate: prior errors from cross-validation (7 folds x 6 seeds), then controller update
    tasks = [(s, i, fo) for s in range(N_SEEDS) for i, fo in enumerate(make_folds(train, N_FOLDS, s))]
    res = Parallel(n_jobs=a.n_jobs)(delayed(_oof_task)(raw, d, table, tm_tab, train, s, i, fo) for s, i, fo in tasks)
    cu = ControllerUpdate(raw)
    rel = []
    for s in range(N_SEEDS):
        oa = pd.concat([r[1] for r in res if r[0] == s]); ob = pd.concat([r[2] for r in res if r[0] == s]).loc[oa.index]
        Pm = np.exp(0.5 * np.log(oa) + 0.5 * np.log(ob)).unstack()[J]
        rel.append((Pm - cu.S.loc[Pm.index, J]) / cu.S.loc[Pm.index, J])
    rel = pd.concat(rel)
    log(f"out-of-fold prior: substrate MAPE {100 * rel.abs().stack().mean():.2f} % (7 folds x {N_SEEDS} seeds)")
    cu.fit(train, rel)
    late, mid = LateDoseRule().fit(raw, train), SubstrateControlRule().fit(raw, train)
    pre = team["substrate"]
    new = pre.copy()
    for b in tpl.batch_id.unique():
        m = (tpl.batch_id == b).to_numpy()
        order = np.argsort(jt[m])
        assert (jt[m][order] == np.arange(1, 14)).all()
        upd = cu.update(b, pre[m][order], late=late, mid=mid)
        nb = np.empty(13); nb[order] = upd
        new[m] = nb
    out["substrate"] = late.apply(pd.DataFrame({"batch_id": tpl.batch_id, "j": jt, "substrate": new})).substrate.to_numpy()
    log("substrate done (controller update + late dose rule)")

    out = out[["batch_id", "elapsed_time", *TARGETS]]
    assert len(out) == len(tpl) and out[TARGETS].notna().all().all() and (out[TARGETS] > 0).all().all()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    log(f"submission written: {a.out}")
    ref = REPO / "submissions" / "blend_base_kalman3_full.csv"
    if ref.exists():
        r = pd.read_csv(ref)
        diff = {v: float((np.abs(out[v].to_numpy() - r[v].to_numpy()) / r[v].to_numpy()).max()) for v in TARGETS}
        log(f"max relative difference vs {ref.name} (LB 7.300): " + ", ".join(f"{v} {x:.1e}" for v, x in diff.items()))


if __name__ == "__main__":
    main()

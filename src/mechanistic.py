"""Mechanistic fed-batch models (ODE + feed boluses) for the DTB challenge.

A model is defined by a `Model` object:
  * its kinetics `rates(x, p)` -> dx/dt, with x[..., :4] = [Xv, Xd, S, P] (+ optional hidden states),
  * its parameter spec (initial guess, bounds, unit, meaning),
  * the concentration of each state in each feed (`feed_conc`),
  * the initial value of hidden states (`init_hidden`).

Between two events the volume is constant (no continuous flow), so concentrations follow dx/dt = rates.
Feed boluses are instantaneous jumps (mass balance):
        V+ = V + dV ,   C+ = (C V + C_feed dV) / V+
At each sampling time the volume is reset to the MEASURED volume (given also for test batches);
withdrawing liquid does not change concentrations.

Models
------
V1 (simple): Monod growth, constant death, Pirt uptake, Luedeking-Piret byproduct.
V2 (from the EDA of apparent specific rates, see notebooks/03):
  growth   mu   = mu_max * S/(Ks+S) * brake            brake = logistic (1 - Xv/Xmax) or hidden nutrient N
  death    kd   = kd0 + kd_S * K_dS/(K_dS + S)          (accelerates when substrate gets low)
  lysis    dead cells disappear at rate k_lys
  uptake   qS   = mu/Yxs + m + q_ovf                    (growth + maintenance + overflow)
  overflow q_ovf= q_ovf_max * S^n/(K_ovf^n + S^n)       (excess uptake at high substrate)
  byproduct qP  = Yps * q_ovf - qPc * P/(K_Pc + P)      (produced by overflow, consumed otherwise)

Numerics: all batches are integrated together with a fixed-step RK4 scheme on a fine grid (dt = 0.01 d);
feed events are applied at the end of the grid step that contains them (timing error <= 0.01 d).
`simulate_ivp` gives a reference solution with scipy's solve_ivp at the exact event times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

import data as D

STATE_NAMES = ["vcd", "dcd", "substrate", "byproduct"]   # observed states (first 4 of the state vector)
FIT_VARS = ["vcd", "dcd", "substrate", "byproduct"]      # variables used in the fit residuals
EPS_S = 0.05  # g/L: smooth switch-off of substrate uptake when substrate is exhausted (avoids S < 0)


@dataclass
class Model:
    name: str
    spec: dict                                 # param -> (init, low, high, unit, meaning)
    rates: Callable                            # (x, p) -> dx/dt
    n_states: int = 4
    feed_conc: Callable = None                 # p -> {feed name: concentration vector (n_states,)}
    init_hidden: Callable = None               # p -> values of hidden states at t=0 (n_states-4,)
    fixed: dict = field(default_factory=dict)  # parameters held constant (not estimated)
    undiluted: tuple = ()                      # indices of hidden states NOT diluted by feeds (history variables)


def _substrate_feeds(p, n_states=4):
    """Default: only the substrate feed and Feed B carry substrate (state index 2)."""
    out = {f: np.zeros(n_states) for f in D.FEEDS}
    out["feed_substrate"][2] = p["C_sub"]
    out["feed_b"][2] = p["C_b"]
    return out


# --------------------------------------------------------------------------
# Version 1: simple model
# --------------------------------------------------------------------------
PARAMS_V1 = {
    "mu_max": (1.0, 0.05, 5.0, "1/d", "maximum specific growth rate"),
    "Ks": (0.5, 1e-3, 20.0, "g/L", "Monod half-saturation constant for substrate"),
    "kd": (0.02, 1e-5, 1.0, "1/d", "specific death rate"),
    "Yxs": (1.5, 0.05, 50.0, "cells/g", "biomass yield on substrate"),
    "m": (0.05, 1e-5, 2.0, "g/(cells d)", "maintenance substrate consumption"),
    "alpha": (0.6, 1e-5, 10.0, "g/cells", "growth-associated byproduct formation (Luedeking-Piret)"),
    "beta": (0.01, 1e-6, 2.0, "g/(cells d)", "non-growth-associated byproduct formation (Luedeking-Piret)"),
    "C_sub": (250.0, 10.0, 3000.0, "g/L", "substrate concentration in the substrate feed"),
    "C_b": (50.0, 0.1, 1000.0, "g/L", "substrate concentration in Feed B"),
}


def rates_v1(x: np.ndarray, p: dict) -> np.ndarray:
    Xv, Xd, S, P = np.moveaxis(x, -1, 0)
    Sp = np.maximum(S, 0.0)
    mu = p["mu_max"] * Sp / (p["Ks"] + Sp)                         # Monod growth
    qS = mu / p["Yxs"] + p["m"] * Sp / (Sp + EPS_S)               # Pirt uptake
    qP = p["alpha"] * mu + p["beta"]                              # Luedeking-Piret
    return np.stack([(mu - p["kd"]) * Xv, p["kd"] * Xv, -qS * Xv, qP * Xv], axis=-1)


V1 = Model("v1 simple", PARAMS_V1, rates_v1, feed_conc=_substrate_feeds)


# --------------------------------------------------------------------------
# Version 2
# --------------------------------------------------------------------------
_COMMON_V2 = {
    "mu_max": (1.0, 0.1, 5.0, "1/d", "maximum specific growth rate"),
    "Ks": (0.1, 1e-3, 5.0, "g/L", "Monod half-saturation constant for substrate (growth)"),
    "kd0": (0.005, 1e-5, 0.5, "1/d", "basal specific death rate"),
    "kd_S": (0.05, 1e-5, 2.0, "1/d", "extra death rate under substrate limitation"),
    "K_dS": (0.5, 1e-2, 10.0, "g/L", "substrate level below which extra death kicks in"),
    "k_lys": (0.1, 1e-4, 5.0, "1/d", "lysis rate of dead cells"),
    "Yxs": (5.0, 0.1, 200.0, "cells/g", "biomass yield on substrate (growth part)"),
    "m": (0.1, 1e-4, 2.0, "g/(cells d)", "maintenance substrate uptake"),
    "q_ovf": (0.6, 1e-3, 5.0, "g/(cells d)", "maximum overflow uptake"),
    "K_ovf": (3.0, 0.3, 10.0, "g/L", "substrate level for half-maximal overflow"),
    "n_ovf": (4.0, 1.0, 12.0, "-", "steepness (Hill coefficient) of the overflow switch"),
    "Yps": (1.0, 0.05, 5.0, "g/g", "byproduct produced per substrate overflowed"),
    "qPc": (0.02, 1e-5, 2.0, "g/(cells d)", "maximum specific byproduct consumption"),
    "K_Pc": (0.5, 1e-3, 10.0, "g/L", "half-saturation of byproduct consumption"),
    "C_sub": (250.0, 10.0, 3000.0, "g/L", "substrate concentration in the substrate feed"),
    "C_b": (50.0, 0.1, 1000.0, "g/L", "substrate concentration in Feed B"),
}


def _v2_core(Xv, Xd, S, P, p, brake):
    """Shared V2 kinetics given a growth brake factor in [0, 1]. Returns (dXv, dXd, dS, dP, mu)."""
    Sp = np.maximum(S, 0.0)
    Pp = np.maximum(P, 0.0)
    fS = Sp / (Sp + EPS_S)                                                     # uptake switch-off at S ~ 0
    mu = p["mu_max"] * Sp / (p["Ks"] + Sp) * brake                             # growth
    kd = p["kd0"] + p["kd_S"] * p["K_dS"] / (p["K_dS"] + Sp)                   # death
    Sn = Sp ** p["n_ovf"]
    q_ovf = p["q_ovf"] * Sn / (p["K_ovf"] ** p["n_ovf"] + Sn)                  # overflow uptake
    qS = mu / p["Yxs"] + (p["m"] + q_ovf) * fS                                 # total uptake
    qP = p["Yps"] * q_ovf * fS - p["qPc"] * Pp / (p["K_Pc"] + Pp)              # byproduct
    return (mu - kd) * Xv, kd * Xv - p["k_lys"] * Xd, -qS * Xv, qP * Xv, mu


# V2a: logistic (cell-density) growth brake
PARAMS_V2_LOGISTIC = {**_COMMON_V2,
                      "Xmax": (32.0, 5.0, 200.0, "1e6 cells/mL", "carrying capacity (logistic brake)")}


def rates_v2_logistic(x, p):
    Xv, Xd, S, P = np.moveaxis(x, -1, 0)
    brake = np.clip(1.0 - Xv / p["Xmax"], 0.0, None)
    dXv, dXd, dS, dP, _ = _v2_core(Xv, Xd, S, P, p, brake)
    return np.stack([dXv, dXd, dS, dP], axis=-1)


V2_LOGISTIC = Model("v2 logistic brake", PARAMS_V2_LOGISTIC, rates_v2_logistic, feed_conc=_substrate_feeds)


# V2b: hidden limiting nutrient N (normalised: N(0) = 1), consumed by growth and supplied by Feed B
PARAMS_V2_NUTRIENT = {**_COMMON_V2,
                      "K_N": (0.1, 1e-3, 10.0, "N0", "half-saturation of growth on the hidden nutrient"),
                      "Yxn": (20.0, 0.1, 1000.0, "cells/N0", "biomass yield on the hidden nutrient"),
                      "C_Nb": (5.0, 1e-3, 1000.0, "N0", "hidden nutrient concentration in Feed B")}


def rates_v2_nutrient(x, p):
    Xv, Xd, S, P, N = np.moveaxis(x, -1, 0)
    Np = np.maximum(N, 0.0)
    brake = Np / (p["K_N"] + Np)
    dXv, dXd, dS, dP, mu = _v2_core(Xv, Xd, S, P, p, brake)
    dN = -mu / p["Yxn"] * Xv
    return np.stack([dXv, dXd, dS, dP, dN], axis=-1)


def _nutrient_feeds(p):
    out = _substrate_feeds(p, n_states=5)
    out["feed_b"][4] = p["C_Nb"]
    return out


def _nutrient_init(p):
    """Hidden nutrient is normalised: N(0) = 1 in every batch."""
    return np.array([1.0])


V2_NUTRIENT = Model("v2 hidden nutrient", PARAMS_V2_NUTRIENT, rates_v2_nutrient, n_states=5,
                    feed_conc=_nutrient_feeds, init_hidden=_nutrient_init)

# --------------------------------------------------------------------------
# Version 3 (built on v2 hidden nutrient; see notebooks/results for the motivation)
#   * overflow switched off by accumulated byproduct ("lactate shift"): q_ovf * K_iP/(K_iP + P)
#     -> lets Feed B carry a realistic substrate load without fake byproduct peaks after each bolus
#   * death no longer driven by low substrate (not supported by between-batch data) but by
#       (H1) depletion of the hidden nutrient (cells die once they can no longer grow):  kd_N * K_dN/(K_dN + N)
#       (H2) accumulated byproduct exposure, with memory (damage state D):              kd_D * D
#            dD/dt = P - k_r * D   (D is a culture-history variable: NOT diluted by feeds)
#     each term can be switched off by the fit (lower bounds close to 0)
#   * optional growth inhibition by the byproduct (Chiu & Du 2025, eq. 18): mu * K_IP/(K_IP + P)
# States: [Xv, Xd, S, P, N, D]
# --------------------------------------------------------------------------
PARAMS_V3 = {
    "mu_max": (5.0, 0.1, 20.0, "1/d", "maximum specific growth rate"),
    "Ks": (4.9, 1e-3, 50.0, "g/L", "Monod half-saturation constant for substrate (growth)"),
    "K_N": (1.78, 1e-3, 10.0, "N0", "half-saturation of growth on the hidden nutrient"),
    "Yxn": (43.6, 0.1, 1000.0, "cells/N0", "biomass yield on the hidden nutrient"),
    "C_Nb": (0.03, 1e-4, 1000.0, "N0", "hidden nutrient concentration in Feed B"),
    "kd0": (0.003, 1e-6, 0.5, "1/d", "basal specific death rate"),
    "kd_N": (0.03, 1e-6, 2.0, "1/d", "extra death rate once the hidden nutrient is depleted (H1)"),
    "K_dN": (0.1, 1e-4, 10.0, "N0", "nutrient level below which H1 death kicks in"),
    "kd_D": (0.001, 1e-7, 1.0, "1/(d * g/L d)", "death rate per unit of accumulated byproduct exposure (H2)"),
    "k_r": (0.05, 1e-4, 10.0, "1/d", "decay ('repair') rate of the byproduct damage"),
    "k_lys": (0.01, 1e-5, 5.0, "1/d", "lysis rate of dead cells"),
    "Yxs": (200.0, 0.1, 1e4, "cells/g", "biomass yield on substrate (growth part)"),
    "m": (0.1, 1e-4, 2.0, "g/(cells d)", "maintenance substrate uptake"),
    "q_ovf": (0.47, 1e-3, 5.0, "g/(cells d)", "maximum overflow uptake"),
    "K_ovf": (3.6, 0.3, 10.0, "g/L", "substrate level for half-maximal overflow"),
    "n_ovf": (8.0, 1.0, 20.0, "-", "steepness (Hill coefficient) of the overflow switch"),
    "K_iP": (0.5, 1e-3, 50.0, "g/L", "byproduct level that halves the overflow (lactate shift)"),
    "Yps": (1.3, 0.05, 5.0, "g/g", "byproduct produced per substrate overflowed"),
    "qPc": (0.1, 1e-5, 2.0, "g/(cells d)", "maximum specific byproduct consumption"),
    "K_Pc": (5.0, 1e-3, 100.0, "g/L", "half-saturation of byproduct consumption"),
    "C_sub": (250.0, 1.0, 3000.0, "g/L", "substrate concentration in the substrate feed"),
    "C_b": (50.0, 0.1, 1000.0, "g/L", "substrate concentration in Feed B"),
}
PARAMS_V3_GI = {**PARAMS_V3,
                "K_IP": (5.0, 1e-2, 1000.0, "g/L", "byproduct level that halves growth (growth inhibition)")}


def _make_rates_v3(growth_inhibition: bool):
    def rates(x, p):
        Xv, Xd, S, P, N, Dm = np.moveaxis(x, -1, 0)
        Sp, Pp = np.maximum(S, 0.0), np.maximum(P, 0.0)
        Np, Dp = np.maximum(N, 0.0), np.maximum(Dm, 0.0)
        fS = Sp / (Sp + EPS_S)                                                     # uptake switch-off at S ~ 0
        mu = p["mu_max"] * Sp / (p["Ks"] + Sp) * Np / (p["K_N"] + Np)              # Monod x hidden nutrient
        if growth_inhibition:
            mu = mu * p["K_IP"] / (p["K_IP"] + Pp)                                 # byproduct inhibition
        kd = (p["kd0"]
              + p["kd_N"] * p["K_dN"] / (p["K_dN"] + Np)                           # H1: nutrient depleted
              + p["kd_D"] * Dp)                                                    # H2: byproduct damage
        Sn = Sp ** p["n_ovf"]
        q_ovf = (p["q_ovf"] * Sn / (p["K_ovf"] ** p["n_ovf"] + Sn)
                 * p["K_iP"] / (p["K_iP"] + Pp))                                   # overflow, shut by P
        qS = mu / p["Yxs"] + (p["m"] + q_ovf) * fS
        qP = p["Yps"] * q_ovf * fS - p["qPc"] * Pp / (p["K_Pc"] + Pp)
        return np.stack([(mu - kd) * Xv,
                         kd * Xv - p["k_lys"] * Xd,
                         -qS * Xv,
                         qP * Xv,
                         -mu / p["Yxn"] * Xv,
                         Pp - p["k_r"] * Dp], axis=-1)
    return rates


def _v3_feeds(p):
    out = _substrate_feeds(p, n_states=6)
    out["feed_b"][4] = p["C_Nb"]
    return out


def _v3_init(p):
    """Hidden nutrient normalised to N(0) = 1; no byproduct damage at inoculation."""
    return np.array([1.0, 0.0])


rates_v3 = _make_rates_v3(False)
rates_v3_gi = _make_rates_v3(True)
V3 = Model("v3", PARAMS_V3, rates_v3, n_states=6, feed_conc=_v3_feeds, init_hidden=_v3_init, undiluted=(5,))
V3_GI = Model("v3 + growth inhibition by byproduct", PARAMS_V3_GI, rates_v3_gi, n_states=6,
              feed_conc=_v3_feeds, init_hidden=_v3_init, undiluted=(5,))

MODELS = {"v1": V1, "v2_logistic": V2_LOGISTIC, "v2_nutrient": V2_NUTRIENT, "v3": V3, "v3_gi": V3_GI}


# --------------------------------------------------------------------------
# Batch preparation: everything the simulator needs, as arrays on a time grid
# --------------------------------------------------------------------------
@dataclass
class BatchSet:
    ids: list[str]
    dt: float
    n_steps: int
    x0: np.ndarray          # (B, 4) initial observed state from the t=0 measurement
    v0: np.ndarray          # (B,) initial volume
    vol_reset: np.ndarray   # (n_steps, B) measured volume imposed at end of step k (NaN = none)
    feed_vol: dict          # feed name -> (n_steps, B) bolus volume added at end of step k
    obs: pd.DataFrame       # one row per sampling time t > 0: batch index, step, weight, measured values


def prepare(raw: pd.DataFrame, batch_ids: list[str], dt: float = 0.01) -> BatchSet:
    """Build the grid representation for the given batches (train or test)."""
    sub = raw[raw["batch_id"].isin(batch_ids)]
    meas = D.measurements(sub, include_to_predict=True)
    feeds = D.feeds_long(sub)
    idx = {b: i for i, b in enumerate(batch_ids)}
    n_steps = int(np.ceil(meas["elapsed_time"].max() / dt)) + 2
    B = len(batch_ids)

    m0 = meas[meas["elapsed_time"] == 0].set_index("batch_id").loc[batch_ids]
    x0 = m0[STATE_NAMES].to_numpy(float)
    v0 = m0["volume"].to_numpy(float)

    # Event at time t is applied at the end of the step k with t in (k*dt, (k+1)*dt]
    step_of = lambda t: np.maximum(np.ceil(np.asarray(t) / dt).astype(int) - 1, 0)

    vol_reset = np.full((n_steps, B), np.nan)
    later = meas[meas["elapsed_time"] > 0]
    vol_reset[step_of(later["elapsed_time"]), later["batch_id"].map(idx)] = later["volume"]

    feed_vol = {}
    for name in D.FEEDS:
        arr = np.zeros((n_steps, B))
        f = feeds[feeds["feed"] == name]
        np.add.at(arr, (step_of(f["elapsed_time"]), f["batch_id"].map(idx).to_numpy()), f["amount"].to_numpy())
        feed_vol[name] = arr

    # Observation: value = (1-w) * state_after_events(k*dt) + w * state_before_events((k+1)*dt)
    obs = later[["batch_id", "elapsed_time", "vcd", "dcd", "tcd", "substrate", "byproduct"]].copy()
    obs["b"] = obs["batch_id"].map(idx)
    obs["k"] = np.floor(obs["elapsed_time"] / dt).astype(int)
    obs["w"] = obs["elapsed_time"] / dt - obs["k"]
    return BatchSet(list(batch_ids), dt, n_steps, x0, v0, vol_reset, feed_vol, obs.reset_index(drop=True))


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def _initial_state(model: Model, x0_obs: np.ndarray, p: dict) -> np.ndarray:
    if model.n_states == 4:
        return x0_obs.copy()
    hidden = np.broadcast_to(model.init_hidden(p), (x0_obs.shape[0], model.n_states - 4))
    return np.hstack([x0_obs, hidden])


def simulate(model: Model, p: dict, bs: BatchSet, return_dense: bool = False):
    """Simulate all batches of `bs`; return predictions at the observation times
    (and optionally the dense state trajectories on the grid, after events)."""
    p = {**p, **model.fixed}
    dt = bs.dt
    x, V = _initial_state(model, bs.x0, p), bs.v0.copy()
    conc = model.feed_conc(p)                                       # feed -> (n_states,)
    post = np.empty((bs.n_steps + 1, *x.shape))                     # after events at grid point k
    pre = np.empty_like(post)                                       # before events at grid point k
    post[0] = pre[0] = x
    for k in range(bs.n_steps):
        k1 = model.rates(x, p); k2 = model.rates(x + 0.5 * dt * k1, p)
        k3 = model.rates(x + 0.5 * dt * k2, p); k4 = model.rates(x + dt * k3, p)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        pre[k + 1] = x
        r = bs.vol_reset[k]
        V = np.where(np.isnan(r), V, r)                             # sampling: impose measured volume
        dVs = {f: bs.feed_vol[f][k] for f in D.FEEDS}
        dV = sum(dVs.values())
        if dV.any():                                                # bolus feeds: dilution + supply
            Vn = V + dV
            added = sum(dVs[f][:, None] * conc[f][None, :] for f in D.FEEDS)
            x_new = (x * V[:, None] + added) / Vn[:, None]
            for j in model.undiluted:                               # history variables are not diluted
                x_new[:, j] = x[:, j]
            x, V = x_new, Vn
        post[k + 1] = x

    o = bs.obs
    w = o["w"].to_numpy()[:, None]
    val = (1 - w) * post[o["k"], o["b"], :4] + w * pre[o["k"] + 1, o["b"], :4]
    out = o[["batch_id", "elapsed_time"]].copy()
    out[STATE_NAMES] = val
    out["tcd"] = out["vcd"] + out["dcd"]
    if return_dense:
        return out, np.arange(bs.n_steps + 1) * dt, post
    return out


def simulate_ivp(model: Model, p: dict, raw: pd.DataFrame, batch_id: str):
    """Reference simulation of one batch with solve_ivp, integrating exactly between events.
    Returns (predictions at sampling times, dense trajectory DataFrame)."""
    p = {**p, **model.fixed}
    g = raw[raw["batch_id"] == batch_id].sort_values("elapsed_time")
    m0 = g[g["elapsed_time"] == 0].iloc[0]
    x0 = m0[["vcd", "tcd", "substrate", "byproduct"]].to_numpy(float)
    x0[1] -= x0[0]                                                  # tcd -> dcd
    x = _initial_state(model, x0[None, :], p)[0]
    V, t, rows, dense = float(m0["volume"]), 0.0, [], []
    conc = model.feed_conc(p)
    fun = lambda _t, y: model.rates(y, p)
    for _, ev in g[g["elapsed_time"] > 0].iterrows():
        te = ev["elapsed_time"]
        sol = solve_ivp(fun, (t, te), x, method="LSODA", rtol=1e-8, atol=1e-10, dense_output=True)
        tt = np.linspace(t, te, 20)
        dense.append(np.c_[tt, sol.sol(tt)[:4].T])
        x, t = sol.y[:, -1].copy(), te
        if ev["row_type"] in ("measurement", "to_predict"):
            rows.append((te, *x[:4]))
            V = ev["volume"]
        else:
            name = next(f for f in D.FEEDS if not np.isnan(ev[f]))
            dV = ev[name]
            x_new = (x * V + conc[name] * dV) / (V + dV)
            for j in model.undiluted:
                x_new[j] = x[j]
            x, V = x_new, V + dV
    pts = pd.DataFrame(rows, columns=["elapsed_time", *STATE_NAMES])
    pts["tcd"] = pts["vcd"] + pts["dcd"]
    return pts, pd.DataFrame(np.vstack(dense), columns=["elapsed_time", *STATE_NAMES])


# --------------------------------------------------------------------------
# Parameter estimation
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    model: Model
    params: dict
    table: pd.DataFrame     # estimate, approx. 95% CI, unit, meaning
    cost: float
    nfev: int
    success: bool


def fit(model: Model, raw: pd.DataFrame, batch_ids: list[str], scales: dict | None = None,
        dt: float = 0.01, init: dict | None = None, max_nfev: int | None = None, verbose: int = 0,
        loss: str = "linear", progress_every: int = 0, tag: str = "",
        residual: str = "scaled", fit_vars: list | None = None, f_scale: float = 1.0) -> FitResult:
    """Least-squares fit of parameters shared by all given (training) batches.

    Residuals = (simulated - measured) / scale for vcd, dcd, substrate, byproduct at every t > 0.
    `scales` defaults to each variable's standard deviation over the training measurements, so the four
    variables weigh comparably. Parameters are estimated in log space (positive, scale-free).
    """
    # residual="scaled": (pred - meas) / std(variable)   -> squared-error type fit
    # residual="log"   : log(pred) - log(meas)            -> relative-error fit, aligned with MAPE
    fv = list(fit_vars or (FIT_VARS if residual == "scaled" else ["vcd", "tcd", "substrate", "byproduct"]))
    bs = prepare(raw, batch_ids, dt)
    y = bs.obs[fv].to_numpy(float)
    if scales is None:
        scales = bs.obs[fv].std().to_dict()
    sc = np.array([scales[v] for v in fv])
    log_y = np.log(np.maximum(y, 1e-6))
    names = [n for n in model.spec if n not in model.fixed]
    init = init or {}
    z0 = np.log([init.get(n, model.spec[n][0]) for n in names])
    lb = np.log([model.spec[n][1] for n in names]); ub = np.log([model.spec[n][2] for n in names])
    z0 = np.clip(z0, lb + 1e-6, ub - 1e-6)

    n_eval = [0]

    def resid(z):
        with np.errstate(all="ignore"):
            pred = simulate(model, dict(zip(names, np.exp(z))), bs)[fv].to_numpy(float)
            if residual == "log":
                r = (np.log(np.maximum(pred, 1e-6)) - log_y).ravel()
            else:
                r = ((pred - y) / sc).ravel()
        r = np.where(np.isfinite(r), r, 1e3)                       # guard against numerical blow-up
        n_eval[0] += 1
        if progress_every and n_eval[0] % progress_every == 0:
            print(f"  {tag} {n_eval[0]} simulations, current SSE/2 = {0.5 * np.sum(r ** 2):.1f}", flush=True)
        return r

    # loss="linear": least squares; loss="soft_l1": robust, close to absolute error for large residuals
    sol = least_squares(resid, z0, bounds=(lb, ub), diff_step=1e-4, max_nfev=max_nfev, verbose=verbose,
                        loss=loss, f_scale=f_scale)

    # Approximate 95% CI from the Jacobian (log-parameters). Residuals within a batch are correlated,
    # so these intervals are optimistic: treat them as indicative only.
    n, k = sol.fun.size, len(names)
    s2 = 2 * sol.cost / max(n - k, 1)
    with np.errstate(all="ignore"):
        try:
            se = np.sqrt(np.diag(np.linalg.pinv(sol.jac.T @ sol.jac) * s2))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)
        table = pd.DataFrame({
            "estimate": np.exp(sol.x),
            "ci95_low": np.exp(sol.x - 1.96 * se),
            "ci95_high": np.exp(np.minimum(sol.x + 1.96 * se, 700)),
            "at_bound": np.isclose(sol.x, lb, atol=1e-3) | np.isclose(sol.x, ub, atol=1e-3),
            "unit": [model.spec[n][3] for n in names],
            "meaning": [model.spec[n][4] for n in names],
        }, index=names)
    return FitResult(model, dict(zip(names, np.exp(sol.x))), table, float(sol.cost), int(sol.nfev), bool(sol.success))


def predict(model: Model, raw: pd.DataFrame, batch_ids: list[str], params: dict, dt: float = 0.01) -> pd.DataFrame:
    """Predict batches from their t=0 state + feed profile (+ given volume) only."""
    return simulate(model, params, prepare(raw, batch_ids, dt))

"""Substrate model for the DTB challenge: exact mass balance + growth kinetics.

Self-contained in new/ (reads the CSVs sitting next to this file).

Idea
----
For a fed-batch with sampling, the substrate mass balance over the interval between
sample j-1 and sample j is EXACT if we use the measured reactor volume:

    S_j * (V_j + r) = S_{j-1} * V_{j-1} + C_sub * FS_j + C_b * FB_j - consumed_j

where
  * V_j is the measured volume at sample j (given for the test batches too, including
    the final one),
  * r is the (constant) sample-withdrawal volume, one global parameter,
  * FS_j / FB_j are the recorded feed_substrate / feed_b bolus volumes in the interval,
  * C_sub / C_b are the substrate concentrations in those feeds (global parameters),
  * consumed_j = integral of qS(S, Xv) * Xv * V dt (growth kinetics below).

The unexplained volume change u_j = V_j - V_{j-1} - F_j mixes the hidden base addition
(pH control, not in the feed records) and the withdrawal; the formula above absorbs
both exactly through the measured V_j -- this is what removes the bias of the
"volume proxy" at the end of the process. Inside the interval the volume profile is
proxied by V_{j-1} + cumulative feeds (+ the unexplained part spread linearly), which
only affects the consumption integral.

Kinetics (states x = [Xv, Xd, S, N], N = hidden limiting nutrient carried by Feed B):
    mu  = mu_max * S/(Ks+S) * N/(K_N+N)          Monod x hidden nutrient (growth)
    kd  = kd0                                    death
    qS  = mu/Yxs + (m + q_ovf(S)) * S/(S+EPS)     specific substrate uptake
    q_ovf(S) = q_ovf_max * S^n / (K_ovf^n + S^n) overflow metabolism at high S
Feed boluses are instantaneous mass additions + dilution: x <- (x V + c dV)/(V + dV).
At each sampling time the state is rescaled by V/(V_j + r) (exact balance above) and
the volume is reset to the measured V_j.

Everything is integrated with a vectorised fixed-step RK4 over all batches at once;
predictions at sample times are read off the grid (timing error <= dt).

Only t=0 measurements + feed profile + volume profile are used to predict a batch.

Usage (from the repo root, python = /home/renku/work/.venv/bin/python):
    python new/substrate_mb.py fitall                       # in-sample floor + params.json
    python new/substrate_mb.py cv --seed 0 --fold 0         # one OOF fold -> new/cv/...
    python new/substrate_mb.py cv --seed 0                  # all 7 folds (serial)
    python new/substrate_mb.py eval                         # aggregate CV MAPE per sample
    python new/substrate_mb.py predict                      # test predictions -> new/submission_substrate_mb.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
EPS_S = 0.05          # g/L, smooth switch-off of uptake at substrate exhaustion
N_OVF = 4.0           # Hill exponent of the overflow switch (fixed)
DT = 0.02             # grid step (days)

# ---------------------------------------------------------------------------
# parameter spec: name -> (init, low, high)
# ---------------------------------------------------------------------------
SPEC = {
    "mu_max": (1.2, 0.05, 8.0),      # 1/d
    "Ks":     (0.5, 1e-3, 20.0),     # g/L
    "K_N":    (0.3, 1e-3, 10.0),     # N0
    "Yxn":    (30.0, 0.1, 1000.0),   # cells/N0
    "C_Nb":   (0.05, 1e-4, 100.0),   # N0/L in feed B
    "kd0":    (0.02, 1e-5, 1.0),     # 1/d
    "klys":   (0.05, 1e-4, 5.0),     # 1/d
    "Yxs":    (3.0, 0.05, 200.0),    # cells/g  (vcd units ~ 1e6 cells/mL vs g/L -> see data)
    "m":      (0.05, 1e-4, 2.0),     # g/(cell d)
    "q_ovf":  (0.5, 1e-4, 5.0),      # g/(cell d)
    "K_ovf":  (3.0, 0.3, 10.0),      # g/L
    "C_sub":  (300.0, 20.0, 3000.0), # g/L in feed_substrate
    "C_b":    (50.0, 0.5, 1000.0),   # g/L in feed_b
    "r":      (0.05, 0.0, 0.3),      # L withdrawn per sample
}
FIXED = {"n_ovf": N_OVF}
N_STATES = 4                      # [Xv, Xd, S, N]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(HERE / "bioreaction_remaining.csv", dtype={"batch_id": str})
    df["to_predict"] = df["to_predict"].astype(bool)
    df["batch_num"] = df["batch_id"].str.extract(r"(\d+)").astype(int)
    is_meas = df[TARGETS].notna().any(axis=1)
    is_feed = df[FEEDS].notna().any(axis=1)
    df["row_type"] = np.select([df["to_predict"], is_meas, is_feed],
                               ["to_predict", "measurement", "feed"], default="unknown")
    return df.sort_values(["batch_num", "elapsed_time"]).reset_index(drop=True)


def split_ids(raw: pd.DataFrame):
    test = sorted(raw.loc[raw.to_predict, "batch_id"].unique(),
                  key=lambda s: int(s[5:]))
    train = sorted(set(raw.batch_id) - set(test), key=lambda s: int(s[5:]))
    return train, test


def make_folds(batch_ids: list[str], n_folds: int = 7, seed: int = 0) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    ids = np.array(batch_ids)[rng.permutation(len(batch_ids))]
    return [sorted(f.tolist(), key=lambda s: int(s[5:]))
            for f in np.array_split(ids, n_folds)]


# ---------------------------------------------------------------------------
# grid representation of a set of batches
# ---------------------------------------------------------------------------
class Grid:
    """Vectorised event grid for a list of batches.

    x0 : (B, 4) initial state [vcd, dcd, substrate, N=1] at t=0
    v0 : (B,) initial measured volume
    feed_vol : {feed: (n_steps, B)} bolus volumes applied at the END of step k
    hidden   : (n_steps, B) unexplained volume (hidden base - withdrawal) spread
               linearly over the interval, applied as dilution at the end of each step
    samp_V   : (n_steps, B) measured volume imposed at the end of step k (NaN = none);
               at those steps the state is rescaled by V/(V_j + r) and V reset to V_j
    obs      : DataFrame(batch_id, b, k, j, elapsed_time, [measured targets or NaN])
               one row per sample with t > 0 (k = step index, prediction read at post[k+1])
    """

    def __init__(self, raw: pd.DataFrame, batch_ids: list[str], dt: float = DT):
        self.dt = dt
        self.ids = list(batch_ids)
        idx = {b: i for i, b in enumerate(batch_ids)}
        B = len(batch_ids)
        sub = raw[raw.batch_id.isin(batch_ids)]
        samp = sub[sub.volume.notna()].sort_values(["batch_num", "elapsed_time"])
        feeds = sub[sub.row_type == "feed"].sort_values(["batch_num", "elapsed_time"])
        t_max = samp.elapsed_time.max()
        self.n_steps = int(np.ceil(t_max / dt)) + 2

        step_of = lambda t: np.maximum(np.ceil(np.asarray(t, dtype=float) / dt).astype(int) - 1, 0)

        # initial state from the t=0 row
        m0 = samp[samp.elapsed_time == 0].set_index("batch_id").loc[batch_ids]
        self.x0 = np.column_stack([m0.vcd, m0.tcd - m0.vcd, m0.substrate,
                                   np.ones(B)]).astype(float)
        self.v0 = m0.volume.to_numpy(float)

        self.feed_vol = {f: np.zeros((self.n_steps, B)) for f in FEEDS}
        for name in FEEDS:
            f = feeds[feeds[name].notna()]
            np.add.at(self.feed_vol[name],
                      (step_of(f.elapsed_time), f.batch_id.map(idx).to_numpy()),
                      f[name].to_numpy())

        # samples with t > 0: measured volume reset + obs rows
        later = samp[samp.elapsed_time > 0]
        self.samp_V = np.full((self.n_steps, B), np.nan)
        ks = step_of(later.elapsed_time)
        self.samp_V[ks, later.batch_id.map(idx).to_numpy()] = later.volume.to_numpy()

        # hidden volume per interval, spread over the steps of the interval
        self.hidden = np.zeros((self.n_steps, B))
        allrows = later[["batch_id", "elapsed_time", "volume"]].copy()
        allrows["b"] = allrows.batch_id.map(idx)
        prev = samp[samp.elapsed_time == 0][["batch_id", "elapsed_time", "volume"]].copy()
        prev["b"] = prev.batch_id.map(idx)
        for _, row in allrows.iterrows():
            p = prev[prev.b == row.b].iloc[0]
            t0, t1 = p.elapsed_time, row.elapsed_time
            F = sum(self.feed_vol[f][step_of(np.array([t0 + 1e-9]))[0]: step_of(np.array([t1]))[0] + 1, row.b].sum()
                    for f in FEEDS)
            u = row.volume - p.volume - F                      # = hidden base - withdrawal
            k0, k1 = step_of(np.array([t0]))[0] + 1, step_of(np.array([t1]))[0]
            self.hidden[k0:k1 + 1, row.b] += u / max(k1 - k0 + 1, 1)

        # observation rows (samples t>0); j = sample index (0 = t=0 row)
        j_of = samp.groupby("batch_id").cumcount()
        later = later.assign(j=j_of[later.index].to_numpy(), b=later.batch_id.map(idx),
                             k=ks)
        self.obs = later[["batch_id", "b", "k", "j", "elapsed_time",
                          "vcd", "tcd", "substrate", "byproduct"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# kinetics
# ---------------------------------------------------------------------------
def rates(x: np.ndarray, p: dict) -> np.ndarray:
    Xv, Xd, S, N = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    Sp = np.maximum(S, 0.0)
    Np = np.maximum(N, 0.0)
    fS = Sp / (Sp + EPS_S)
    mu = p["mu_max"] * Sp / (p["Ks"] + Sp) * Np / (p["K_N"] + Np)
    kd = p["kd0"]
    Sn = Sp ** FIXED["n_ovf"]
    q_ovf = p["q_ovf"] * Sn / (p["K_ovf"] ** FIXED["n_ovf"] + Sn)
    qS = mu / p["Yxs"] + (p["m"] + q_ovf) * fS
    dXv = (mu - kd) * Xv
    dXd = kd * Xv - p["klys"] * Xd
    dS = -qS * Xv
    dN = -mu / p["Yxn"] * Xv
    return np.stack([dXv, dXd, dS, dN], axis=1)


def feed_conc(p: dict) -> dict:
    z4 = np.zeros(N_STATES)
    out = {f: z4.copy() for f in FEEDS}
    out["feed_substrate"][2] = p["C_sub"]
    out["feed_b"][2] = p["C_b"]
    out["feed_b"][3] = p["C_Nb"]
    return out


# ---------------------------------------------------------------------------
# simulation (vectorised RK4)
# ---------------------------------------------------------------------------
def simulate(g: Grid, p: dict) -> pd.DataFrame:
    dt = g.dt
    x = g.x0.copy()
    V = g.v0.copy()
    conc = feed_conc(p)
    r = p["r"]
    post = np.empty((g.n_steps + 1, x.shape[0], N_STATES))
    post[0] = x
    for k in range(g.n_steps):
        k1 = rates(x, p)
        k2 = rates(x + 0.5 * dt * k1, p)
        k3 = rates(x + 0.5 * dt * k2, p)
        k4 = rates(x + dt * k3, p)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        # hidden (unexplained) volume: pure dilution
        h = g.hidden[k]
        if h.any():
            m = h > 0
            x[m] = x[m] * (V[m, None] / (V[m] + h[m])[:, None])
            V = V + h
        # feed boluses: mass addition + dilution
        dV = np.zeros_like(V)
        added = np.zeros_like(x)
        for f in FEEDS:
            dv = g.feed_vol[f][k]
            if dv.any():
                dV += dv
                added += dv[:, None] * conc[f][None, :]
        if dV.any():
            Vn = V + dV
            x = (x * V[:, None] + added) / Vn[:, None]
            V = Vn
        # sampling: exact mass balance with the measured volume V_j
        s = g.samp_V[k]
        m = ~np.isnan(s)
        if m.any():
            x[m] = x[m] * (V[m, None] / (s[m] + r)[:, None])
            V = np.where(m, s, V)
        post[k + 1] = x
    o = g.obs
    val = post[o.k.to_numpy() + 1, o.b.to_numpy(), :]
    out = o[["batch_id", "elapsed_time", "j"]].copy()
    out["vcd"] = val[:, 0]
    out["tcd"] = val[:, 0] + val[:, 1]
    out["substrate"] = val[:, 2]
    out["byproduct"] = np.nan
    return out


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def fit_params(raw: pd.DataFrame, batch_ids: list[str], wS: float = 1.0, wX: float = 0.5,
               dt: float = DT, max_nfev: int = 120, init: dict | None = None,
               verbose: int = 0) -> tuple[dict, dict]:
    """Least-squares fit of the global parameters on the given (training) batches.

    Residuals are log-scale (relative) errors on substrate (weight wS) and on
    vcd / tcd (weight wX) at every sample t > 0 -> aligned with the MAPE metric.
    """
    g = Grid(raw, batch_ids, dt)
    o = g.obs  # simulate() returns rows in exactly this order
    log_y = {v: np.log(o[v].to_numpy(float)) for v in ("substrate", "vcd", "tcd")}

    names = list(SPEC)
    z0 = np.log([(init or {}).get(n, SPEC[n][0]) for n in names])
    lb = np.log([SPEC[n][1] for n in names])
    ub = np.log([SPEC[n][2] for n in names])
    z0 = np.clip(z0, lb + 1e-9, ub - 1e-9)

    def resid(z):
        p = dict(zip(names, np.exp(z)))
        pred = simulate(g, p)
        rS = np.log(np.maximum(pred.substrate.to_numpy(float), 1e-6)) - log_y["substrate"]
        rX = np.log(np.maximum(pred.vcd.to_numpy(float), 1e-6)) - log_y["vcd"]
        rT = np.log(np.maximum(pred.tcd.to_numpy(float), 1e-6)) - log_y["tcd"]
        r = np.concatenate([wS * rS, wX * rX, wX * rT])
        return np.where(np.isfinite(r), r, 1e3)

    sol = least_squares(resid, z0, bounds=(lb, ub), diff_step=1e-4,
                        max_nfev=max_nfev, verbose=verbose)
    params = dict(zip(names, np.exp(sol.x)))
    info = {"cost": float(sol.cost), "nfev": int(sol.nfev), "success": bool(sol.success),
            "names": names, "x": sol.x.tolist()}
    return params, info


def predict(raw: pd.DataFrame, batch_ids: list[str], params: dict, dt: float = DT) -> pd.DataFrame:
    return simulate(Grid(raw, batch_ids, dt), params)


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
def mape_by_j(pred: pd.DataFrame, raw: pd.DataFrame, var: str = "substrate") -> pd.DataFrame:
    m = raw[(raw.row_type == "measurement")][["batch_id", "elapsed_time", var]].copy()
    k = lambda d: d.assign(_t=d.elapsed_time.round(6))
    p = k(pred)[["batch_id", "_t", "j", var]].merge(
        k(m)[["batch_id", "_t", var]], on=["batch_id", "_t"], suffixes=("_pred", ""))
    p["ape"] = 100 * np.abs(p[f"{var}_pred"] - p[var]) / p[var]
    per_j = p.groupby("j").ape.agg(["mean", "count"])
    return p, per_j


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_fitall(args):
    raw = load_raw()
    train, test = split_ids(raw)
    params, info = fit_params(raw, train, wS=args.ws, wX=args.wx,
                              max_nfev=args.max_nfev, verbose=2)
    out = HERE / "params_mb.json"
    out.write_text(json.dumps({"params": params, "info": {k: v for k, v in info.items() if k != "x"}},
                              indent=2))
    print(json.dumps(params, indent=2))
    # in-sample floor
    pred = predict(raw, train, params)
    p, per_j = mape_by_j(pred, raw)
    print(f"in-sample substrate MAPE: {p.ape.mean():.3f}%")
    print(per_j.round(3).to_string())


def cmd_cv(args):
    raw = load_raw()
    train, test = split_ids(raw)
    folds = make_folds(train, 7, args.seed)
    outdir = HERE / "cv"
    outdir.mkdir(exist_ok=True)
    todo = range(7) if args.fold is None else [args.fold]
    for f in todo:
        val = folds[f]
        tr = [b for b in train if b not in val]
        params, info = fit_params(raw, tr, wS=args.ws, wX=args.wx, max_nfev=args.max_nfev)
        pred = predict(raw, val, params)
        p, per_j = mape_by_j(pred, raw)
        pred.to_csv(outdir / f"oof_s{args.seed}_f{f}.csv", index=False)
        print(f"seed {args.seed} fold {f}: substrate MAPE {p.ape.mean():.3f}% "
              f"(cost {info['cost']:.1f}, nfev {info['nfev']})", flush=True)
        print(per_j["mean"].round(3).to_string(), flush=True)


def cmd_eval(args):
    raw = load_raw()
    import glob
    rows = []
    for f in sorted(glob.glob(str(HERE / "cv" / "oof_s*_f*.csv"))):
        seed = int(f.split("_s")[1].split("_")[0])
        oof = pd.read_csv(f)
        p, per_j = mape_by_j(oof, raw)
        for j, r in per_j.iterrows():
            rows.append(dict(seed=seed, j=j, mape=r["mean"]))
        rows.append(dict(seed=seed, j="ALL", mape=p.ape.mean()))
    df = pd.DataFrame(rows)
    piv = df.pivot(index="j", columns="seed", values="mape")
    piv["mean_all"] = piv.mean(axis=1)
    piv["mean_0-2"] = piv[[0, 1, 2]].mean(axis=1)
    piv["mean_3-5"] = piv[[3, 4, 5]].mean(axis=1) if all(s in piv.columns for s in (3, 4, 5)) else np.nan
    print(piv.round(3).to_string())


def cmd_predict(args):
    raw = load_raw()
    train, test = split_ids(raw)
    pj = HERE / "params_mb.json"
    if pj.exists() and not args.refit:
        params = json.loads(pj.read_text())["params"]
    else:
        params, info = fit_params(raw, train, wS=args.ws, wX=args.wx, max_nfev=args.max_nfev)
        pj.write_text(json.dumps({"params": params}, indent=2))
    pred = predict(raw, test, params)
    tpl = pd.read_csv(HERE / "bioreaction_template.csv", dtype={"batch_id": str})
    k = lambda d: d.assign(_t=d.elapsed_time.round(6))
    out = k(tpl).merge(k(pred)[["batch_id", "_t", "vcd", "tcd", "substrate"]],
                       on=["batch_id", "_t"], how="left").drop(columns="_t")
    assert out.substrate.notna().all() and (out.substrate > 0).all()
    out["byproduct"] = 0.0  # placeholder, filled by the caller from the existing blend
    out = out[tpl.columns]
    out.to_csv(HERE / "submission_substrate_mb.csv", index=False)
    print("wrote new/submission_substrate_mb.csv", out.shape)
    print(out.groupby("batch_id").substrate.apply(lambda s: s.round(3).tolist()).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fitall", "cv", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--fold", type=int, default=None)
        p.add_argument("--ws", type=float, default=1.0, help="weight of substrate residuals")
        p.add_argument("--wx", type=float, default=0.5, help="weight of vcd/tcd residuals")
        p.add_argument("--max_nfev", type=int, default=120)
        p.add_argument("--refit", action="store_true")
    sub.add_parser("eval")
    args = ap.parse_args()
    {"fitall": cmd_fitall, "cv": cmd_cv, "eval": cmd_eval, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()

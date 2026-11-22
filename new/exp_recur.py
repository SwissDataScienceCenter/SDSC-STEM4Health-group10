"""Joint fit: exact mass-balance one-step model with kinetics consumption.

log S1 = log( S0*V0 + C_sub*FS + C_b*FB - consumed ) - log( V1 + r )
consumed = ( mu(S0)/Yxs + (m + q_ovf(S0)) * S0/(S0+EPS) ) * Xavg * Vavg * dt

All params (C_sub, C_b, r, mu_max, Ks, Yxs, m, q_ovf, K_ovf) fitted jointly on
train intervals, minimizing log-residuals of S1 (teacher-forced S0, measured X).
Then recursive prediction (S0 from t=0 measurement only, X from ODE or measured).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import substrate_mb as M

EPS = 0.05
PNAMES = ["C_sub", "C_b", "r", "mu_max", "Ks", "Yxs", "m", "q_ovf", "K_ovf"]
P0 =    [250.0, 100.0, 0.05, 1.0, 0.5, 3.0, 0.05, 0.3, 3.0]
PLO =   [20.0, 0.5, 0.0, 0.05, 1e-3, 0.05, 1e-4, 1e-4, 0.3]
PHI =   [3000.0, 1000.0, 0.3, 8.0, 20.0, 200.0, 2.0, 5.0, 10.0]


def intervals(raw, batch_ids):
    m = raw[(raw.row_type == "measurement") & raw.batch_id.isin(batch_ids)].copy()
    m["j"] = m.groupby("batch_id").cumcount()
    m = m.sort_values(["batch_id", "elapsed_time"]).reset_index(drop=True)
    rows = []
    for b, g in m.groupby("batch_id"):
        g = g.sort_values("elapsed_time").reset_index(drop=True)
        allg = raw[raw.batch_id == b].sort_values("elapsed_time")
        for i in range(1, len(g)):
            t0, t1 = g.elapsed_time[i - 1], g.elapsed_time[i]
            win = allg[(allg.elapsed_time > t0) & (allg.elapsed_time <= t1)]
            rows.append(dict(batch_id=b, j=i, deltat=t1 - t0, V0=g.volume[i - 1], V1=g.volume[i],
                             S0=g.substrate[i - 1], S1=g.substrate[i],
                             X0=g.vcd[i - 1], X1=g.vcd[i],
                             FS=win.feed_substrate.sum(), FB=win.feed_b.sum(),
                             FC=win.feed_c.sum(), FD=win.feed_d.sum()))
    return pd.DataFrame(rows).reset_index(drop=True)


def consumed(iv, p):
    S = np.maximum(iv.S0.to_numpy(), 0.0)
    mu = p["mu_max"] * S / (p["Ks"] + S)
    Sn = S ** 4
    q_ovf = p["q_ovf"] * Sn / (p["K_ovf"] ** 4 + Sn)
    qS = mu / p["Yxs"] + (p["m"] + q_ovf) * S / (S + EPS)
    X = 0.5 * (iv.X0.to_numpy() + iv.X1.to_numpy())
    V = 0.5 * (iv.V0.to_numpy() + iv.V1.to_numpy())
    return qS * X * V * iv.deltat.to_numpy()


def s1_hat(iv, p, S0=None):
    S0 = iv.S0.to_numpy() if S0 is None else S0
    cons = consumed(iv, p)
    return (S0 * iv.V0.to_numpy() + p["C_sub"] * iv.FS.to_numpy() + p["C_b"] * iv.FB.to_numpy()
            - cons) / (iv.V1.to_numpy() + p["r"])


def fit(raw, train):
    iv = intervals(raw, train)
    logy = np.log(iv.S1.to_numpy())

    def resid(z):
        p = dict(zip(PNAMES, np.exp(z)))
        return np.log(np.maximum(s1_hat(iv, p), 1e-6)) - logy

    z0 = np.clip(np.log(P0), np.log(PLO) + 1e-9, np.log(PHI) - 1e-9)
    sol = least_squares(resid, z0, bounds=(np.log(PLO), np.log(PHI)), max_nfev=400,
                        diff_step=1e-4)
    p = dict(zip(PNAMES, np.exp(sol.x)))
    return iv, p, sol


def report(iv, p, tag):
    h = s1_hat(iv, p)
    ape = 100 * np.abs(h - iv.S1.to_numpy()) / iv.S1.to_numpy()
    print(f"[{tag}] one-step MAPE {ape.mean():.3f}")
    print("  per j:", {int(k): round(v, 2) for k, v in pd.Series(ape, index=iv.j).groupby(level=0).mean().items()})
    # recursive
    rec = np.full(len(iv), np.nan)
    cons = consumed(iv, p)
    for b, g in iv.groupby("batch_id"):
        g = g.sort_values("j")
        S = g.S0.iloc[0]
        for name, row in g.iterrows():
            S = (S * row.V0 + p["C_sub"] * row.FS + p["C_b"] * row.FB - cons[name]) / (row.V1 + p["r"])
            rec[name] = S
    ape_r = 100 * np.abs(rec - iv.S1.to_numpy()) / iv.S1.to_numpy()
    print(f"[{tag}] recursive MAPE {ape_r.mean():.3f}")
    print("  per j:", {int(k): round(v, 2) for k, v in pd.Series(ape_r, index=iv.j).groupby(level=0).mean().items()})
    return ape, ape_r


if __name__ == "__main__":
    raw = M.load_raw()
    train, test = M.split_ids(raw)
    iv, p, sol = fit(raw, train)
    print("fitted params:", {k: round(v, 4) for k, v in p.items()}, "cost", round(sol.cost, 2), "nfev", sol.nfev)
    report(iv, p, "joint kin, measured X")

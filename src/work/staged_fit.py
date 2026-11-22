"""Staged estimation of global ODE params, then least_squares polish.

Stage A: mu(t) from spline derivatives of log vcd; grid-search (Ks, Kx, KP)
         and linear-solve mu = mu_max*f(S)*g(X) - kd0 - kd1*P/(KP+P).
Stage B: interval substrate mass balances -> OLS for (1/Yxs, ms, g_fs, g_fb, g_fc, g_fd).
Stage C: interval byproduct mass balances -> grid KPc, OLS for (alpha, beta, qc).
Stage D: full least_squares polish from staged x0 (log-ratio residuals, clipped).
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import SimSet, load_batches, PNAMES, PBOUNDS, P0, mape_from_preds


def interp_log(t, y, tt):
    y = np.maximum(y, 1e-6)
    return np.exp(UnivariateSpline(t, np.log(y), k=3, s=max(len(t) - 8, 1) * 0.01)(tt))


def stageABC(B, obs_ids, dt=0.005):
    rows_mu = []
    for b in obs_ids:
        bd = B[b]
        t, Y = bd["t_obs"], bd["y_obs"]
        tt = np.linspace(0.15, t[-1] - 0.15, 80)
        spl = UnivariateSpline(t, np.log(np.maximum(Y[:, 0], 1e-6)), k=3, s=max(len(t) - 8, 1) * 0.01)
        mu = spl.derivative()(tt)
        S = interp_log(t, Y[:, 2], tt)
        P = interp_log(t, Y[:, 3], tt)
        X = np.exp(spl(tt))
        for a in range(len(tt)):
            rows_mu.append([mu[a], S[a], P[a], X[a]])
    R = np.array(rows_mu)
    mu, S, P, X = R[:, 0], R[:, 1], R[:, 2], R[:, 3]

    best = None
    for Ks in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0]:
        fS = S / (Ks + S)
        for Kx in [20, 24, 26, 28, 30, 34, 40]:
            gX = np.clip(1 - X / Kx, 0, None)
            for KP in [0.3, 0.5, 1.0, 2.0, 5.0, 20.0]:
                fP = P / (KP + P)
                A = np.column_stack([fS * gX, -np.ones_like(mu), -fP])
                coef, *_ = np.linalg.lstsq(A, mu, rcond=None)
                r = A @ coef - mu
                cost = (r ** 2).sum()
                if best is None or cost < best[0]:
                    best = (cost, Ks, Kx, KP, coef)
    cost, Ks, Kx, KP, coef = best
    mu_max, kd0, kd1 = coef
    print(f"stage A: Ks={Ks} Kx={Kx} KP={KP} mu_max={mu_max:.3f} kd0={kd0:.4f} kd1={kd1:.4f} rms={np.sqrt(cost/len(mu)):.3f}")

    # interval balances for S and P using model mu on obs grid
    dFS, dFB, dFC, dFD = [], [], [], []
    iMuX, iX, iXP = [], [], []
    dMS, dMP = [], []
    for b in obs_ids:
        bd = B[b]
        t, Y = bd["t_obs"], bd["y_obs"]
        Xv, S, P = Y[:, 0], Y[:, 2], Y[:, 3]
        vt, vv = bd["v_t"], bd["v_v"]
        V = np.interp(t, vt, vv)
        ft, fa = bd["f_t"], bd["f_a"]
        cum = np.array([fa[ft <= tt + 1e-9].sum(0) for tt in t])
        for i in range(len(t) - 1):
            dt_ = t[i + 1] - t[i]
            Xm = 0.5 * (Xv[i] + Xv[i + 1]); Vm = 0.5 * (V[i] + V[i + 1])
            Sm = 0.5 * (S[i] + S[i + 1]); Pm = 0.5 * (P[i] + P[i + 1])
            mu_m = mu_max * Sm / (Ks + Sm) * max(0.0, 1 - Xm / Kx) - (kd0 + kd1 * Pm / (KP + Pm))
            iMuX.append(mu_m * Xm * Vm * dt_)
            iX.append(Xm * Vm * dt_)
            iXP.append(Xm * Vm * Pm / (0.5 + Pm) * dt_)  # KPc grid below uses 0.5 placeholder
            dMS.append(S[i + 1] * V[i + 1] - S[i] * V[i])
            dMP.append(P[i + 1] * V[i + 1] - P[i] * V[i])
            dFS.append(cum[i + 1, 0] - cum[i, 0]); dFB.append(cum[i + 1, 1] - cum[i, 1])
            dFC.append(cum[i + 1, 2] - cum[i, 2]); dFD.append(cum[i + 1, 3] - cum[i, 3])
    iMuX, iX, dMS, dMP = map(np.array, (iMuX, iX, dMS, dMP))
    dF = np.column_stack([dFS, dFB, dFC, dFD])

    # stage B: dMS = -(1/Yxs)*iMuX - ms*iX + dF@g
    A = np.column_stack([-iMuX, -iX, dF])
    coef, *_ = np.linalg.lstsq(A, dMS, rcond=None)
    invY, ms = coef[0], coef[1]
    g = coef[2:]
    Yxs = 1.0 / invY if invY > 0 else 100.0
    r = A @ coef - dMS
    print(f"stage B: Yxs={Yxs:.3f} ms={ms:.4f} gains g_fs={g[0]:.2f} g_fb={g[1]:.2f} g_fc={g[2]:.2f} g_fd={g[3]:.2f} rms={r.std():.3f}")

    # stage C: dMP = alpha*iMuX + beta*iX - qc*iXP(KPc) ; grid KPc
    best = None
    for KPc in [0.1, 0.2, 0.3, 0.5, 0.8, 1.5, 3.0, 10.0]:
        # recompute iXP with this KPc: need P per interval; redo quickly
        iXP2 = []
        for b in obs_ids:
            bd = B[b]
            t, Y = bd["t_obs"], bd["y_obs"]
            Xv, P = Y[:, 0], Y[:, 3]
            vt, vv = bd["v_t"], bd["v_v"]
            V = np.interp(t, vt, vv)
            for i in range(len(t) - 1):
                dt_ = t[i + 1] - t[i]
                Xm = 0.5 * (Xv[i] + Xv[i + 1]); Vm = 0.5 * (V[i] + V[i + 1])
                Pm = 0.5 * (P[i] + P[i + 1])
                iXP2.append(Xm * Vm * Pm / (KPc + Pm) * dt_)
        iXP2 = np.array(iXP2)
        A = np.column_stack([iMuX, iX, -iXP2])
        coef, *_ = np.linalg.lstsq(A, dMP, rcond=None)
        r = A @ coef - dMP
        cost = (r ** 2).sum()
        if best is None or cost < best[0]:
            best = (cost, KPc, coef)
    cost, KPc, coef = best
    alpha, beta, qc = coef
    print(f"stage C: KPc={KPc} alpha={alpha:.4f} beta={beta:.4f} qc={qc:.4f} rms={np.sqrt(cost/len(dMP)):.3f}")

    p0 = P0.copy()
    p0[0] = mu_max; p0[1] = Ks; p0[2] = Kx; p0[3] = 50.0  # KiP off
    p0[4] = max(kd0, 1e-4); p0[5] = max(kd1, 0.0); p0[6] = KP
    p0[7] = Yxs; p0[8] = max(ms, 1e-4); p0[9] = max(alpha, 0.0); p0[10] = max(beta, 0.0)
    p0[11] = max(qc, 0.0); p0[12] = KPc
    p0[13:17] = np.maximum(g, 0.0)
    return p0


def polish(B, obs_ids, p0, dt=0.005, nfev=600, free=None):
    sim = SimSet(B, obs_ids, dt=dt)
    obs_y, obs_n, obs_idx = sim.obs_y, sim.obs_n, sim.obs_idx
    if free is None:
        free = list(range(21))

    def residual(x):
        logp = np.log(np.maximum(p0, 1e-9)).copy()
        logp[free] = x
        p2 = np.exp(logp)[None, :].repeat(sim.n, axis=0)
        M_all = sim.simulate(p2)
        res = []
        for j in range(sim.n):
            m = obs_n[j]
            if m <= 1:
                continue
            idx = obs_idx[j, 1:m]
            pred = M_all[idx, j, :] / sim.V[idx, j, None]
            pred = np.maximum(pred, 1e-4)
            yo = obs_y[j, 1:m]
            res.append(np.log(pred / yo).ravel())
        return np.concatenate(res)

    lo = np.log(np.maximum(PBOUNDS[free, 0], 1e-9))
    hi = np.log(PBOUNDS[free, 1])
    x0 = np.clip(np.log(np.maximum(p0[free], 1e-9)), lo + 1e-9, hi - 1e-9)
    t0 = time.time()
    out = least_squares(residual, x0, bounds=(lo, hi), method="trf",
                        x_scale="jac", max_nfev=nfev)
    logp = np.log(np.maximum(p0, 1e-9)).copy()
    logp[free] = out.x
    p2 = np.exp(logp)[None, :].repeat(sim.n, axis=0)
    preds = sim.obs_preds(p2)
    m = mape_from_preds(preds, sim, skip_first=True)
    print(f"polish: cost {out.cost:.3f} nfev {out.nfev} ({time.time()-t0:.0f}s) "
          f"in-sample MAPE {np.round(m,2)} mean {m.mean():.2f}")
    return logp, m


if __name__ == "__main__":
    B, tpl, df = load_batches()
    obs_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    p0 = stageABC(B, obs_ids)
    print("staged p0:", {PNAMES[i]: round(p0[i], 4) for i in range(21)})
    logp, m = polish(B, obs_ids, p0, dt=0.005, nfev=800)
    print("final params:", {PNAMES[i]: round(float(np.exp(logp[i])), 5) for i in range(21)})
    json.dump({"logp": logp.tolist(), "params": {PNAMES[i]: float(np.exp(logp[i])) for i in range(21)},
               "mape": m.tolist()}, open(os.path.join(os.path.dirname(__file__), "params_staged.json"), "w"), indent=1)

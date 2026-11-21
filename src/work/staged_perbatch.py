"""Data-driven staged init + coupled polish per batch.

Stage 1 (from data, smoothing splines):
  kd(t) = dM_Xd/dt / M_Xv ;  mu(t) = dln(M_Xv)/dt + kd(t)
  grid (Ks, Kx, KP) + OLS -> mu_max ; OLS kd0, kd1
  substrate balance OLS -> (1/Yxs, ms, gains) ; byproduct grid KPc + OLS -> (alpha, beta, qc)
Stage 2: coupled polish (block descent growth/metabolites + full), multi-start.

Usage: python staged_perbatch.py [--dt 0.005] [--out staged_perbatch.json] [--only ...]
"""
import sys, os, json, time, argparse
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import SimSet, load_batches, PNAMES, PBOUNDS, P0, mape_from_preds
from coupled_perbatch import BlockFit, GIDX, MIDX, base_logp


def smooth_deriv(t, y, tt, s_rel=0.02):
    y = np.maximum(y, 1e-9)
    spl = UnivariateSpline(t, y, k=3, s=max(len(t) - 8, 1) * s_rel)
    return spl(tt), spl.derivative()(tt)


def stage1(B, b):
    bd = B[b]
    t, Y = bd["t_obs"], bd["y_obs"]
    Xv, Xd, S, P = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
    V = np.interp(t, bd["v_t"], bd["v_v"])
    M_Xv = Xv * V
    M_Xd = Xd * V
    tt = np.linspace(0.1, t[-1] - 0.05, 60)
    Vt = np.interp(tt, bd["v_t"], bd["v_v"])
    M_Xv_s, dM_Xv = smooth_deriv(t, M_Xv, tt)
    _, dM_Xd = smooth_deriv(t, M_Xd, tt)
    M_Xv_t = np.interp(tt, t, M_Xv)
    kd = np.clip(dM_Xd / np.maximum(M_Xv_t, 1e-9), 0, None)
    mu = dM_Xv / np.maximum(M_Xv_t, 1e-9) + kd
    S_t = np.exp(UnivariateSpline(t, np.log(np.maximum(S, 1e-4)), k=3,
                                   s=max(len(t) - 8, 1) * 0.02)(tt))
    P_t = np.exp(UnivariateSpline(t, np.log(np.maximum(P, 1e-4)), k=3,
                                   s=max(len(t) - 8, 1) * 0.02)(tt))
    Xv_t = M_Xv_t / Vt
    # growth: mu = mu_max * S/(Ks+S) * clip(1-Xv/Kx,0) ; grid Ks,Kx
    best = None
    for Ks in [0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0]:
        for Kx in [20, 22, 24, 26, 28, 31, 35]:
            f = S_t / (Ks + S_t) * np.clip(1 - Xv_t / Kx, 0, None)
            A = f[:, None]
            coef, *_ = np.linalg.lstsq(A, mu, rcond=None)
            r = A @ coef - mu
            c = (r ** 2).sum()
            if best is None or c < best[0]:
                best = (c, Ks, Kx, coef[0])
    c, Ks, Kx, mu_max = best
    # death: kd = kd0 + kd1*P/(KP+P) ; grid KP
    best = None
    for KP in [0.1, 0.3, 0.6, 1.0, 2.0, 5.0, 20.0]:
        A = np.column_stack([np.ones_like(P_t), P_t / (KP + P_t)])
        coef, *_ = np.linalg.lstsq(A, kd, rcond=None)
        r = A @ coef - kd
        c = (r ** 2).sum()
        if best is None or c < best[0]:
            best = (c, KP, coef)
    c, KP, (kd0, kd1) = best
    # substrate balance on intervals: dM_S = -(mu/Y+ms) M_Xv dt + g.dF
    ft, fa = bd["f_t"], bd["f_a"]
    rows = []
    for i in range(len(t) - 1):
        dt_ = t[i + 1] - t[i]
        M_Xv_m = 0.5 * (M_Xv[i] + M_Xv[i + 1])
        mu_m = np.interp(0.5 * (t[i] + t[i + 1]), tt, mu)
        dF = fa[(ft > t[i]) & (ft <= t[i + 1] + 1e-9)].sum(0) if ((ft > t[i]) & (ft <= t[i + 1] + 1e-9)).any() else np.zeros(4)
        dMS = S[i + 1] * V[i + 1] - S[i] * V[i]
        rows.append([mu_m * M_Xv_m * dt_, M_Xv_m * dt_, dF[0], dF[1], dF[2], dF[3], dMS])
    Rm = np.array(rows)
    A = np.column_stack([-Rm[:, 0], -Rm[:, 1], Rm[:, 2], Rm[:, 3], Rm[:, 4], Rm[:, 5]])
    coef, *_ = np.linalg.lstsq(A, Rm[:, 6], rcond=None)
    invY, ms, g_fs, g_fb, g_fc, g_fd = coef
    Yxs = 1.0 / invY if invY > 1e-6 else 100.0
    # byproduct balance: dM_P = (alpha*mu+beta) M_Xv dt - qc M_Xv P/(KPc+P) dt
    rows = []
    for i in range(len(t) - 1):
        dt_ = t[i + 1] - t[i]
        M_Xv_m = 0.5 * (M_Xv[i] + M_Xv[i + 1])
        mu_m = np.interp(0.5 * (t[i] + t[i + 1]), tt, mu)
        P_m = 0.5 * (P[i] + P[i + 1])
        dMP = P[i + 1] * V[i + 1] - P[i] * V[i]
        rows.append([mu_m * M_Xv_m * dt_, M_Xv_m * dt_, M_Xv_m * P_m / (0.5 + P_m) * dt_, dMP])
    Rp = np.array(rows)
    best = None
    for KPc in [0.1, 0.2, 0.3, 0.5, 0.8, 1.5, 3.0, 10.0]:
        Kcol = np.array([0.5 * (M_Xv[i] * V[i] * P[i] / (KPc + P[i]) + M_Xv[i + 1] * V[i + 1] * P[i + 1] / (KPc + P[i + 1])) * (t[i + 1] - t[i])
                          for i in range(len(t) - 1)])
        A = np.column_stack([Rp[:, 0], Rp[:, 1], -Kcol])
        coef, *_ = np.linalg.lstsq(A, Rp[:, 3], rcond=None)
        r = A @ coef - Rp[:, 3]
        c = (r ** 2).sum()
        if best is None or c < best[0]:
            best = (c, KPc, coef)
    c, KPc, (alpha, beta, qc) = best
    p = np.maximum(P0, 1e-9).copy()
    p[0] = np.clip(mu_max, 0.05, 3); p[1] = Ks; p[2] = Kx; p[3] = 1e9
    p[4] = max(kd0, 1e-5); p[5] = max(kd1, 0.0); p[6] = KP
    p[7] = np.clip(Yxs, 0.5, 200); p[8] = max(ms, 1e-5)
    p[9] = max(alpha, 0.0); p[10] = max(beta, 0.0); p[11] = max(qc, 0.0); p[12] = KPc
    p[13] = max(g_fs, 0.0); p[14] = max(g_fb, 0.0); p[15] = max(g_fc, 0.0); p[16] = max(g_fd, 0.0)
    return np.log(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--nfev", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "staged_perbatch.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    B, tpl, df = load_batches()
    obs_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    if args.only:
        obs_ids = [b for b in obs_ids if b in args.only.split(",")]
    rng = np.random.default_rng(0)
    results = {}
    if args.resume and os.path.exists(args.out):
        results = json.load(open(args.out))
    t0 = time.time()
    for b in obs_ids:
        if b in results:
            continue
        logp0 = stage1(B, b)
        sim = SimSet(B, [b], dt=args.dt)
        bf = BlockFit(sim)
        logp = logp0.copy()
        for r in range(args.rounds):
            logp, _ = bf.fit_block(logp, GIDX, (0, 1), args.starts, args.nfev, rng)
            logp, _ = bf.fit_block(logp, MIDX, (2, 3), args.starts, args.nfev, rng)
        logp, ca = bf.fit_all(logp, starts=args.starts, nfev=args.nfev, rng=rng)
        m = mape_from_preds(sim.obs_preds(np.exp(logp)[None, :]), sim)
        results[b] = {"logp": logp.tolist(),
                      "params": {PNAMES[i]: float(np.exp(logp[i])) for i in range(21)},
                      "mape": m.tolist(), "cost": float(ca)}
        print(f"{b}: MAPE {np.round(m,2)} mean {m.mean():.2f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(results, open(args.out, "w"))
        t0 = time.time()
    allm = np.array([results[b]["mape"] for b in results])
    print("staged per-batch MAPE per-target:", np.round(allm.mean(0), 3),
          "mean", round(float(allm.mean()), 3))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

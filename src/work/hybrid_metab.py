"""Hybrid metabolite model: S and P ODEs driven by a GIVEN cell trajectory.

    dM_S/dt = -(mu(S)/Y + ms) * M_Xv(t) + feed impulses
    dM_P/dt = (alpha*mu(S) + beta) * M_Xv(t) - qc * M_Xv(t) * P/(KPc+P)
    mu(S) = mu_max * S/(Ks+S) * max(0, 1 - Xv/Kx)

M_Xv(t) comes from observed vcd (train) or predicted vcd (hidden), interpolated
densely. Pooled least_squares over 13 params, numba-vectorized over batches.

Usage: python hybrid_metab.py [--nfev 300] [--starts 6] [--dt 0.005] [--out hybrid_params.json]
"""
import sys, os, json, time, argparse
import numpy as np
from scipy.optimize import least_squares
from numba import njit

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import load_batches

DT = 0.005
T_END = 13.2
# params: mu_max, Ks, Kx, Y, ms, alpha, beta, qc, KPc, g_fs, g_fb, g_fc, g_fd
HP = ["mu_max", "Ks", "Kx", "Y", "ms", "alpha", "beta", "qc", "KPc",
      "g_fs", "g_fb", "g_fc", "g_fd"]
HBOUNDS = np.array([(0.05, 5.0), (0.01, 20.0), (10.0, 100.0), (0.5, 200.0),
                    (1e-5, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 5.0), (0.02, 20.0),
                    (0.0, 5000.0), (0.0, 5000.0), (0.0, 50000.0), (0.0, 50000.0)])
H0 = np.array([1.2, 1.0, 30.0, 8.0, 0.02, 0.1, 0.02, 0.1, 0.5,
               100.0, 40.0, 0.0, 0.0])


@njit(cache=True)
def _metab_sim(p, MXv, V, raw, M_S0, M_P0, dt, n_steps):
    """p: (n,13); MXv,V: (n_steps+1, n); raw: (n_steps, n, 4); M_S0/M_P0: (n,)
    Returns MS_all, MP_all: (n_steps+1, n)."""
    n = MXv.shape[1]
    MS = M_S0.copy()
    MP = M_P0.copy()
    MS_all = np.zeros((n_steps + 1, n))
    MP_all = np.zeros((n_steps + 1, n))
    MS_all[0] = MS
    MP_all[0] = MP
    for k in range(n_steps):
        for j in range(n):
            if raw[k, j, 0] != 0.0 or raw[k, j, 1] != 0.0 or raw[k, j, 2] != 0.0 or raw[k, j, 3] != 0.0:
                MS[j] += p[j, 9] * raw[k, j, 0] + p[j, 10] * raw[k, j, 1] + p[j, 11] * raw[k, j, 2] + p[j, 12] * raw[k, j, 3]
        dS0 = np.zeros(n); dP0 = np.zeros(n)
        dS1 = np.zeros(n); dP1 = np.zeros(n)
        dS2 = np.zeros(n); dP2 = np.zeros(n)
        dS3 = np.zeros(n); dP3 = np.zeros(n)
        for stage in range(4):
            if stage == 0:
                MSk, MPk, Vk, MXk = MS, MP, V[k], MXv[k]
            elif stage == 1:
                MSk = MS_all[k] + 0.5 * dt * dS0
                MPk = MP_all[k] + 0.5 * dt * dP0
                Vk = 0.5 * (V[k] + V[k + 1]); MXk = 0.5 * (MXv[k] + MXv[k + 1])
            elif stage == 2:
                MSk = MS_all[k] + 0.5 * dt * dS1
                MPk = MP_all[k] + 0.5 * dt * dP1
                Vk = 0.5 * (V[k] + V[k + 1]); MXk = 0.5 * (MXv[k] + MXv[k + 1])
            else:
                MSk = MS_all[k] + dt * dS2
                MPk = MP_all[k] + dt * dP2
                Vk = V[k + 1]; MXk = MXv[k + 1]
            dS = np.zeros(n); dP = np.zeros(n)
            for j in range(n):
                S = MSk[j] / Vk[j]
                P = MPk[j] / Vk[j]
                Xv = MXk[j] / Vk[j]
                mu_max, Ks, Kx, Y, ms, alpha, beta, qc, KPc = p[j, 0], p[j, 1], p[j, 2], p[j, 3], p[j, 4], p[j, 5], p[j, 6], p[j, 7], p[j, 8]
                cr = 1.0 - Xv / Kx
                if cr < 0.0:
                    cr = 0.0
                mu = mu_max * S / (Ks + S) * cr
                dS[j] = -(mu / Y + ms) * MXk[j]
                dP[j] = (alpha * mu + beta) * MXk[j] - qc * MXk[j] * P / (KPc + P)
            if stage == 0:
                dS0, dP0 = dS, dP
            elif stage == 1:
                dS1, dP1 = dS, dP
            elif stage == 2:
                dS2, dP2 = dS, dP
            else:
                dS3, dP3 = dS, dP
        for j in range(n):
            MS[j] = MS_all[k, j] + (dt / 6.0) * (dS0[j] + 2 * dS1[j] + 2 * dS2[j] + dS3[j])
            MP[j] = MP_all[k, j] + (dt / 6.0) * (dP0[j] + 2 * dP1[j] + 2 * dP2[j] + dP3[j])
            MS_all[k + 1, j] = MS[j]
            MP_all[k + 1, j] = MP[j]
    return MS_all, MP_all


class MetabSim:
    def __init__(self, B, ids, vcd_override=None, dt=DT, t_end=T_END):
        self.dt = float(dt)
        self.n_steps = int(round(t_end / dt))
        self.ids = list(ids)
        n = self.n = len(ids)
        tg = np.arange(self.n_steps + 1) * self.dt
        self.V = np.zeros((self.n_steps + 1, n))
        self.MXv = np.zeros((self.n_steps + 1, n))
        self.raw = np.zeros((self.n_steps, n, 4))
        self.M_S0 = np.zeros(n)
        self.M_P0 = np.zeros(n)
        self.obs_idx = np.zeros((n, 20), dtype=np.int64)
        self.obs_n = np.zeros(n, dtype=np.int64)
        self.obs_y = np.zeros((n, 20, 2))  # S, P
        self.pred_idx = np.zeros((n, 20), dtype=np.int64)
        self.pred_n = np.zeros(n, dtype=np.int64)
        for j, b in enumerate(ids):
            bd = B[b]
            t, Y = bd["t_obs"], bd["y_obs"]
            self.V[:, j] = np.interp(tg, bd["v_t"], bd["v_v"])
            if vcd_override is not None and b in vcd_override:
                # vcd_override[b]: (len(t_obs)+..., ) values aligned with sorted row times
                tv, xv = vcd_override[b]
                self.MXv[:, j] = np.interp(tg, tv, xv) * self.V[:, j]
            else:
                self.MXv[:, j] = np.interp(tg, t, Y[:, 0]) * self.V[:, j]
            ft, fa = bd["f_t"], bd["f_a"]
            for tf, a in zip(ft, fa):
                k = int(round(tf / self.dt))
                if 0 <= k < self.n_steps:
                    self.raw[k, j] += a
            self.M_S0[j] = Y[0, 2] * bd["v_v"][0]
            self.M_P0[j] = Y[0, 3] * bd["v_v"][0]
            no = len(t)
            self.obs_idx[j, :no] = np.clip(np.round(t / self.dt).astype(int), 0, self.n_steps)
            self.obs_n[j] = no
            self.obs_y[j, :no, 0] = Y[:, 2]
            self.obs_y[j, :no, 1] = Y[:, 3]
            tp = bd["t_pred"]
            if tp is not None and len(tp):
                self.pred_n[j] = len(tp)
                self.pred_idx[j, :len(tp)] = np.clip(np.round(tp / self.dt).astype(int), 0, self.n_steps)

    def residual(self, logp):
        p = np.exp(logp)[None, :].repeat(self.n, axis=0)
        MS_all, MP_all = _metab_sim(p, self.MXv, self.V, self.raw, self.M_S0, self.M_P0,
                                    self.dt, self.n_steps)
        res = []
        for j in range(self.n):
            m = self.obs_n[j]
            if m <= 1:
                continue
            idx = self.obs_idx[j, 1:m]
            predS = np.maximum(MS_all[idx, j] / self.V[idx, j], 1e-4)
            predP = np.maximum(MP_all[idx, j] / self.V[idx, j], 1e-4)
            yo = self.obs_y[j, 1:m]
            res.append(np.log(predS / yo[:, 0]))
            res.append(np.log(predP / yo[:, 1]))
        return np.concatenate(res)

    def mape(self, logp):
        p = np.exp(logp)[None, :].repeat(self.n, axis=0)
        MS_all, MP_all = _metab_sim(p, self.MXv, self.V, self.raw, self.M_S0, self.M_P0,
                                    self.dt, self.n_steps)
        num = np.zeros(2); den = np.zeros(2)
        for j in range(self.n):
            m = self.obs_n[j]
            if m <= 1:
                continue
            idx = self.obs_idx[j, 1:m]
            predS = MS_all[idx, j] / self.V[idx, j]
            predP = MP_all[idx, j] / self.V[idx, j]
            yo = self.obs_y[j, 1:m]
            num[0] += np.abs(predS - yo[:, 0]).sum(); den[0] += np.abs(yo[:, 0]).sum()
            num[1] += np.abs(predP - yo[:, 1]).sum(); den[1] += np.abs(yo[:, 1]).sum()
        return 100 * num / np.maximum(den, 1e-12)

    def predict(self, logp, which="pred"):
        p = np.exp(logp)[None, :].repeat(self.n, axis=0)
        MS_all, MP_all = _metab_sim(p, self.MXv, self.V, self.raw, self.M_S0, self.M_P0,
                                    self.dt, self.n_steps)
        out = []
        for j in range(self.n):
            idx = self.pred_idx[j, : self.pred_n[j]] if which == "pred" else self.obs_idx[j, : self.obs_n[j]]
            if len(idx) == 0:
                out.append(None)
                continue
            out.append(np.column_stack([MS_all[idx, j] / self.V[idx, j],
                                        MP_all[idx, j] / self.V[idx, j]]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfev", type=int, default=300)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "hybrid_params.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    B, tpl, df = load_batches()
    obs_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    sim = MetabSim(B, obs_ids, dt=args.dt)
    lo = np.log(np.maximum(HBOUNDS[:, 0], 1e-12)); hi = np.log(HBOUNDS[:, 1])
    rng = np.random.default_rng(args.seed)
    best = None
    t0 = time.time()
    for s in range(args.starts + 1):
        x0 = np.log(np.maximum(H0, 1e-9)) if s == 0 else rng.uniform(lo, hi)
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        out = least_squares(sim.residual, x0, bounds=(lo, hi), method="trf",
                            x_scale="jac", max_nfev=args.nfev, ftol=1e-12, xtol=1e-12)
        m = sim.mape(out.x)
        print(f"start {s}: cost {out.cost:.3f} nfev {out.nfev} S/P MAPE {np.round(m,2)} ({time.time()-t0:.0f}s)", flush=True)
        if best is None or out.cost < best[0]:
            best = (out.cost, out.x)
    cost, logp = best
    m = sim.mape(logp)
    print("BEST cost", round(cost, 3), "S/P MAPE", np.round(m, 3), "mean", round(m.mean(), 3))
    print("params:", {HP[i]: round(float(np.exp(logp[i])), 5) for i in range(13)})
    json.dump({"logp": logp.tolist(), "params": {HP[i]: float(np.exp(logp[i])) for i in range(13)},
               "mape": m.tolist()}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

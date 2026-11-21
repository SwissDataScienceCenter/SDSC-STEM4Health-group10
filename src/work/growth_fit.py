"""Decoupled growth fit: Xv, Xd ODE driven by OBSERVED S(t), P(t), V(t).

    dM_Xv/dt = (mu - kd) M_Xv
    dM_Xd/dt = kd M_Xv
    mu = mu_max * S/(Ks+S) * KiP/(KiP+P) * max(0, 1 - Xv/Kx)   [KiP optional]
    kd = kd0 + kd1 * P/(KP+P)

Params (7): mu_max, Ks, Kx, KiP, kd0, kd1, KP.  Pooled multi-start over batches.
"""
import sys, os, json, time
import numpy as np
from scipy.optimize import least_squares
from numba import njit

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import load_batches

DT = 0.005
T_END = 13.2
GP = ["mu_max", "Ks", "Kx", "KiP", "kd0", "kd1", "KP"]
GBOUNDS = np.array([(0.05, 5.0), (0.01, 20.0), (10.0, 100.0), (0.05, 100.0),
                    (1e-5, 1.0), (1e-5, 3.0), (0.02, 50.0)])
G0 = np.array([1.0, 0.5, 30.0, 100.0, 0.05, 0.1, 1.0])  # KiP init high (off)


@njit(cache=True)
def _growth_sim(p, M0, V, Sg, Pg, dt, n_steps):
    """p: (n,7); M0: (n,2); V,Sg,Pg: (n_steps+1, n). Returns M_all (n_steps+1, n, 2)."""
    n = M0.shape[0]
    M = M0.copy()
    M_all = np.zeros((n_steps + 1, n, 2))
    M_all[0] = M
    k1 = np.zeros((n, 2)); k2 = np.zeros((n, 2)); k3 = np.zeros((n, 2)); k4 = np.zeros((n, 2))
    for k in range(n_steps):
        for stage in range(4):
            if stage == 0:
                MM = M; Vk = V[k]; Sk = Sg[k]; Pk = Pg[k]
            elif stage == 1:
                MM = M_all[k] + 0.5 * dt * k1; Vk = 0.5*(V[k]+V[k+1]); Sk = 0.5*(Sg[k]+Sg[k+1]); Pk = 0.5*(Pg[k]+Pg[k+1])
            elif stage == 2:
                MM = M_all[k] + 0.5 * dt * k2; Vk = 0.5*(V[k]+V[k+1]); Sk = 0.5*(Sg[k]+Sg[k+1]); Pk = 0.5*(Pg[k]+Pg[k+1])
            else:
                MM = M_all[k] + dt * k3; Vk = V[k+1]; Sk = Sg[k+1]; Pk = Pg[k+1]
            out = k1 if stage == 0 else (k2 if stage == 1 else (k3 if stage == 2 else k4))
            for j in range(n):
                Xv = MM[j, 0] / Vk[j]
                S = Sk[j]; P = Pk[j]
                mu_max, Ks, Kx, KiP, kd0, kd1, KP = p[j, 0], p[j, 1], p[j, 2], p[j, 3], p[j, 4], p[j, 5], p[j, 6]
                cr = 1.0 - Xv / Kx
                if cr < 0.0:
                    cr = 0.0
                mu = mu_max * S / (Ks + S) * KiP / (KiP + P) * cr
                kd = kd0 + kd1 * P / (KP + P)
                out[j, 0] = (mu - kd) * MM[j, 0]
                out[j, 1] = kd * MM[j, 0]
        for j in range(n):
            for c in range(2):
                M[j, c] = M_all[k, j, c] + (dt / 6.0) * (k1[j, c] + 2*k2[j, c] + 2*k3[j, c] + k4[j, c])
            M_all[k+1, j, 0] = M[j, 0]
            M_all[k+1, j, 1] = M[j, 1]
    return M_all


class GrowthSim:
    def __init__(self, B, ids, dt=DT, t_end=T_END):
        self.dt = float(dt)
        self.n_steps = int(round(t_end / dt))
        self.ids = list(ids)
        n = self.n = len(ids)
        tg = np.arange(self.n_steps + 1) * self.dt
        self.M0 = np.zeros((n, 2))
        self.V = np.zeros((self.n_steps + 1, n))
        self.Sg = np.zeros((self.n_steps + 1, n))
        self.Pg = np.zeros((self.n_steps + 1, n))
        self.obs_idx = np.zeros((n, 20), dtype=np.int64)
        self.obs_n = np.zeros(n, dtype=np.int64)
        self.obs_y = np.zeros((n, 20, 2))
        for j, b in enumerate(ids):
            bd = B[b]
            t, Y = bd["t_obs"], bd["y_obs"]
            self.M0[j, 0] = Y[0, 0] * bd["v_v"][0]
            self.M0[j, 1] = Y[0, 1] * bd["v_v"][0]
            self.V[:, j] = np.interp(tg, bd["v_t"], bd["v_v"])
            # log-interp of observed S, P (clip negatives)
            for col, arr in ((2, self.Sg), (3, self.Pg)):
                y = np.maximum(Y[:, col], 1e-4)
                from scipy.interpolate import UnivariateSpline
                if len(t) >= 4:
                    spl = UnivariateSpline(t, np.log(y), k=3, s=max(len(t)-8, 1)*0.02)
                    arr[:, j] = np.exp(spl(tg))
                else:
                    arr[:, j] = np.interp(tg, t, y)
            no = len(t)
            self.obs_idx[j, :no] = np.clip(np.round(t / self.dt).astype(int), 0, self.n_steps)
            self.obs_n[j] = no
            self.obs_y[j, :no, 0] = Y[:, 0]
            self.obs_y[j, :no, 1] = Y[:, 1]

    def residual(self, logp):
        p = np.exp(logp)[None, :].repeat(self.n, axis=0)
        M_all = _growth_sim(p, self.M0, self.V, self.Sg, self.Pg, self.dt, self.n_steps)
        res = []
        for j in range(self.n):
            m = self.obs_n[j]
            if m <= 1:
                continue
            idx = self.obs_idx[j, 1:m]
            pred = M_all[idx, j, :] / self.V[idx, j, None]
            pred = np.maximum(pred, 1e-4)
            yo = self.obs_y[j, 1:m]
            res.append(np.log(pred / yo).ravel())
        return np.concatenate(res)

    def mape(self, logp):
        p = np.exp(logp)[None, :].repeat(self.n, axis=0)
        M_all = _growth_sim(p, self.M0, self.V, self.Sg, self.Pg, self.dt, self.n_steps)
        num = np.zeros(2); den = np.zeros(2)
        for j in range(self.n):
            m = self.obs_n[j]
            if m <= 1:
                continue
            idx = self.obs_idx[j, 1:m]
            pred = M_all[idx, j, :] / self.V[idx, j, None]
            yo = self.obs_y[j, 1:m]
            num += np.abs(pred - yo).sum(0)
            den += np.abs(yo).sum(0)
        return 100 * num / np.maximum(den, 1e-12)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--nfev", type=int, default=300)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--kip", action="store_true", help="free KiP (lactate inhibition)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "growth_params.json"))
    args = ap.parse_args()

    B, tpl, df = load_batches()
    obs_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    sim = GrowthSim(B, obs_ids, dt=args.dt)
    lo = np.log(GBOUNDS[:, 0]); hi = np.log(GBOUNDS[:, 1])
    free = [0, 1, 2, 4, 5, 6] + ([3] if args.kip else [])
    rng = np.random.default_rng(0)
    best = None
    t0 = time.time()
    for s in range(args.starts + 1):
        if s == 0:
            x0full = np.log(G0)
        else:
            x0full = rng.uniform(lo, hi)
        x0 = x0full[free]
        lof, hif = lo[free], hi[free]
        x0 = np.clip(x0, lof + 1e-9, hif - 1e-9)

        def res(x):
            lf = np.log(G0).copy()
            lf[free] = x
            return sim.residual(lf)
        out = least_squares(res, x0, bounds=(lof, hif), method="trf", x_scale="jac",
                            max_nfev=args.nfev, ftol=1e-12, xtol=1e-12)
        lf = np.log(G0).copy(); lf[free] = out.x
        m = sim.mape(lf)
        print(f"start {s}: cost {out.cost:.3f} nfev {out.nfev} vcd/xd MAPE {np.round(m,2)} ({time.time()-t0:.0f}s)", flush=True)
        if best is None or out.cost < best[0]:
            best = (out.cost, lf)
    cost, lf = best
    m = sim.mape(lf)
    print("BEST cost", round(cost, 4), "MAPE vcd,xd", np.round(m, 3), "mean", round(m.mean(), 3))
    print("growth params:", {GP[i]: round(float(np.exp(lf[i])), 5) for i in range(7)})
    json.dump({"logp": lf.tolist(), "params": {GP[i]: float(np.exp(lf[i])) for i in range(7)},
               "mape": m.tolist()}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

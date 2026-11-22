"""Per-batch COUPLED ODE fit via block coordinate descent + multi-start.

Blocks: growth params G = [mu_max, Ks, Kx, KiP, kd0, kd1, KP] (residuals: vcd, xd)
        metabolite params M = [Yxs, ms, alpha, beta, qc, KPc, g_fs, g_fb, g_fc, g_fd]
                            (residuals: substrate, byproduct)
Alternate least_squares on each block (multi-start), then a full polish.

Usage: python coupled_perbatch.py [--dt 0.005] [--rounds 3] [--out coupled_perbatch.json]
                                  [--only Batch1,Batch2] [--resume]
"""
import sys, os, json, time
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import SimSet, load_batches, PNAMES, PBOUNDS, P0, mape_from_preds

GIDX = [0, 1, 2, 3, 4, 5, 6]          # mu_max, Ks, Kx, KiP, kd0, kd1, KP
MIDX = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Yxs, ms, alpha, beta, qc, KPc, gains


def base_logp():
    p = np.maximum(P0, 1e-9).copy()
    p[3] = 1e9   # KiP off initially
    p[17:21] = 0.0
    return np.log(p)


class BlockFit:
    def __init__(self, sim):
        self.sim = sim
        self.obs_y = sim.obs_y
        self.obs_n = sim.obs_n
        self.obs_idx = sim.obs_idx

    def residual(self, logp, cols):
        p2 = np.exp(logp)[None, :].repeat(self.sim.n, axis=0)
        M_all = self.sim.simulate(p2)
        res = []
        for j in range(self.sim.n):
            m = self.obs_n[j]
            if m <= 1:
                continue
            idx = self.obs_idx[j, 1:m]
            pred = M_all[idx, j, :] / self.sim.V[idx, j, None]
            pred = np.maximum(pred, 1e-4)
            yo = self.obs_y[j, 1:m]
            res.append(np.log(pred[:, cols] / yo[:, cols]).ravel())
        return np.concatenate(res)

    def fit_block(self, logp, idxs, cols, starts, nfev, rng):
        lo = np.log(np.maximum(PBOUNDS[idxs, 0], 1e-12))
        hi = np.log(PBOUNDS[idxs, 1])

        def res(x):
            lf = logp.copy()
            lf[idxs] = x
            return self.residual(lf, cols)
        best = None
        for s in range(starts):
            x0 = logp[idxs] if s == 0 else rng.uniform(lo, hi)
            x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
            out = least_squares(res, x0, bounds=(lo, hi), method="trf", x_scale="jac",
                                max_nfev=nfev, ftol=1e-12, xtol=1e-12)
            if best is None or out.cost < best[0]:
                best = (out.cost, out.x)
        logp = logp.copy()
        logp[idxs] = best[1]
        return logp, best[0]

    def fit_all(self, logp, starts, nfev, rng, idxs=None, cols=(0, 1, 2, 3)):
        if idxs is None:
            idxs = list(range(21))
        lo = np.log(np.maximum(PBOUNDS[idxs, 0], 1e-12))
        hi = np.log(PBOUNDS[idxs, 1])

        def res(x):
            lf = logp.copy()
            lf[idxs] = x
            return self.residual(lf, cols)
        best = None
        for s in range(starts):
            x0 = logp[idxs] if s == 0 else rng.uniform(lo, hi)
            x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
            out = least_squares(res, x0, bounds=(lo, hi), method="trf", x_scale="jac",
                                max_nfev=nfev, ftol=1e-12, xtol=1e-12)
            if best is None or out.cost < best[0]:
                best = (out.cost, out.x)
        logp = logp.copy()
        logp[idxs] = best[1]
        return logp, best[0]


def fit_one_batch(B, b, dt, rounds, starts, nfev, rng):
    sim = SimSet(B, [b], dt=dt)
    bf = BlockFit(sim)
    logp = base_logp()
    for r in range(rounds):
        logp, cg = bf.fit_block(logp, GIDX, (0, 1), starts, nfev, rng)
        logp, cm = bf.fit_block(logp, MIDX, (2, 3), starts, nfev, rng)
    logp, ca = bf.fit_all(logp, starts=max(2, starts // 2), nfev=nfev, rng=rng)
    p2 = np.exp(logp)[None, :]
    m = mape_from_preds(sim.obs_preds(p2), sim)
    return logp, m, ca


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--nfev", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "coupled_perbatch.json"))
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
        logp, m, ca = fit_one_batch(B, b, args.dt, args.rounds, args.starts, args.nfev, rng)
        results[b] = {"logp": logp.tolist(),
                      "params": {PNAMES[i]: float(np.exp(logp[i])) for i in range(21)},
                      "mape": m.tolist(), "cost": float(ca)}
        print(f"{b}: MAPE {np.round(m,2)} mean {m.mean():.2f} cost {ca:.3f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(results, open(args.out, "w"))
        t0 = time.time()
    allm = np.array([results[b]["mape"] for b in results])
    print("coupled per-batch MAPE per-target:", np.round(allm.mean(0), 3),
          "overall mean-of-means", round(float(allm.mean()), 3))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

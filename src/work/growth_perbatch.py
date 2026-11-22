"""Per-batch decoupled growth fits with multi-start; store params + meta-features.

Usage: python growth_perbatch.py [--starts 10] [--nfev 400] [--dt 0.005] [--kip]
"""
import sys, os, json, time
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from growth_fit import GrowthSim, GBOUNDS, G0, GP
from ode_numba import load_batches


def fit_batch(sim, free, lo, hi, starts, nfev, rng):
    def res(x):
        lf = np.log(G0).copy()
        lf[free] = x
        return sim.residual(lf)
    best = None
    for s in range(starts):
        x0 = np.log(G0)[free] if s == 0 else rng.uniform(lo[free], hi[free])
        x0 = np.clip(x0, lo[free] + 1e-9, hi[free] - 1e-9)
        out = least_squares(res, x0, bounds=(lo[free], hi[free]), method="trf",
                            x_scale="jac", max_nfev=nfev, ftol=1e-12, xtol=1e-12)
        if best is None or out.cost < best[0]:
            lf = np.log(G0).copy(); lf[free] = out.x
            best = (out.cost, lf, out.nfev)
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--nfev", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--kip", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "growth_perbatch.json"))
    args = ap.parse_args()

    B, tpl, df = load_batches()
    obs_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    hid_ids = sorted([b for b in B if B[b]["hidden"]], key=lambda s: int(s[5:]))
    lo = np.log(GBOUNDS[:, 0]); hi = np.log(GBOUNDS[:, 1])
    free = [0, 1, 2, 4, 5, 6] + ([3] if args.kip else [])
    rng = np.random.default_rng(0)

    results = {}
    t0 = time.time()
    for b in obs_ids:
        sim = GrowthSim(B, [b], dt=args.dt)
        cost, lf, nfev = fit_batch(sim, free, lo, hi, args.starts, args.nfev, rng)
        m = sim.mape(lf)
        results[b] = {"logp": lf.tolist(), "params": {GP[i]: float(np.exp(lf[i])) for i in range(7)},
                      "mape": m.tolist(), "cost": float(cost), "nfev": int(nfev)}
        print(f"{b}: vcd/xd MAPE {np.round(m,2)} cost {cost:.3f} nfev {nfev} "
              f"mu_max={np.exp(lf[0]):.3f} Ks={np.exp(lf[1]):.2f} Kx={np.exp(lf[2]):.1f} "
              f"kd0={np.exp(lf[4]):.4f} kd1={np.exp(lf[5]):.4f} KP={np.exp(lf[6]):.2f} ({time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()
    json.dump(results, open(args.out, "w"), indent=1)
    allm = np.array([results[b]["mape"] for b in results])
    print("per-batch vcd/xd MAPE: mean of means", round(float(allm.mean()), 3),
          "per-target", np.round(allm.mean(0), 3))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

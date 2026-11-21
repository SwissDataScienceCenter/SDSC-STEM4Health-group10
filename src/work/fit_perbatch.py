"""Per-batch ODE fits (structure check + per-batch params for meta-regression).

Usage: python fit_perbatch.py [--nfev 300] [--dt 0.005] [--out perbatch_params.json]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import SimSet, load_batches, PNAMES, PBOUNDS, P0, mape_from_preds

# free per-batch params (subset of the 21): indices into PNAMES
FREE = [0, 1, 2, 4, 5, 7, 9, 10]  # mu_max, Ks, Kx, kd0, kd1, Yxs, alpha, beta
FIXED = {i: P0[i] for i in range(21) if i not in FREE}
# tighter per-batch bounds
FB = {
    0: (0.05, 3.0), 1: (0.02, 10.0), 2: (5.0, 200.0), 4: (1e-5, 0.5),
    5: (1e-5, 2.0), 7: (0.5, 200.0), 9: (0.0, 2.0), 10: (0.0, 2.0),
}


def make_residual(sim, global_logp):
    obs_y, obs_n, obs_idx = sim.obs_y, sim.obs_n, sim.obs_idx

    def residual(x):
        # x: free log-params for ALL batches in sim (n, len(FREE))
        logp = np.tile(global_logp, (sim.n, 1))
        for a, i in enumerate(FREE):
            logp[:, i] = x[:, a]
        p2 = np.exp(logp)
        M_all = sim.simulate(p2)
        res = []
        for j in range(sim.n):
            m = obs_n[j]
            if m <= 1:
                continue
            idx = obs_idx[j, 1:m]
            pred = M_all[idx, j, :] / sim.V[idx, j, None]
            yo = obs_y[j, 1:m]
            r = np.log((pred + 1e-6) / (yo + 1e-6))
            res.append(r.ravel())
        return np.concatenate(res)
    return residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfev", type=int, default=250)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--global-path", dest="global_path", default=os.path.join(os.path.dirname(__file__), "params_global.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "perbatch_params.json"))
    ap.add_argument("--only", default=None, help="comma-separated batch ids")
    args = ap.parse_args()

    B, tpl, df = load_batches()
    obs_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    if args.only:
        obs_ids = [b for b in obs_ids if b in args.only.split(",")]

    if os.path.exists(args.global_path):
        g = json.load(open(args.global_path))
        global_logp = np.array(g["logp"])
    else:
        global_logp = np.log(np.maximum(P0, 1e-6))

    results = {}
    t_all = time.time()
    for b in obs_ids:
        sim = SimSet(B, [b], dt=args.dt)
        res_fn = make_residual(sim, global_logp)
        x0 = np.tile(np.log(np.maximum(P0[FREE], 1e-6)), (1, 1))[0]
        lo = np.log([FB[i][0] if FB[i][0] > 0 else 1e-9 for i in FREE])
        hi = np.log([FB[i][1] for i in FREE])
        out = least_squares(lambda x: res_fn(x[None, :]), x0, bounds=(lo, hi),
                            method="trf", x_scale="jac", max_nfev=args.nfev)
        logp_b = global_logp.copy()
        for a, i in enumerate(FREE):
            logp_b[i] = out.x[a]
        p2 = np.exp(logp_b)[None, :]
        preds = sim.obs_preds(p2)
        m = mape_from_preds(preds, sim, skip_first=True)
        results[b] = dict(logp=logp_b.tolist(), params={PNAMES[i]: float(np.exp(logp_b[i])) for i in range(21)},
                          mape=m.tolist(), mape_mean=float(m.mean()), cost=float(out.cost), nfev=int(out.nfev))
        print(f"{b}: MAPE {np.round(m,2)} mean {m.mean():.2f} cost {out.cost:.3f} nfev {out.nfev} "
              f"({time.time()-t_all:.0f}s)", flush=True)
        t_all = time.time()
    json.dump(results, open(args.out, "w"), indent=1)
    allm = np.array([results[b]["mape"] for b in results])
    print("per-batch MAPE mean over batches:", np.round(allm.mean(0), 3), "overall", round(float(allm.mean()), 3))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

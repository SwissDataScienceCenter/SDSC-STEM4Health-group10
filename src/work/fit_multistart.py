"""Multi-start global ODE fit. Usage:
python fit_multistart.py [--starts 24] [--nfev 250] [--dt 0.01] [--subset 20]
                          [--out params_ms.json] [--folds 0]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import SimSet, load_batches, PNAMES, PBOUNDS, P0, mape_from_preds

# free params: everything except KiP (idx 3, fixed OFF) and h_* (idx 17-20, fixed 0)
FREE = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
FIXED = {3: 1e9, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0}


def base_logp():
    p = np.maximum(P0, 1e-9).copy()
    for i, v in FIXED.items():
        p[i] = v
    return np.log(p)


def make_residual(sim):
    obs_y, obs_n, obs_idx = sim.obs_y, sim.obs_n, sim.obs_idx

    def residual(x):
        logp = base_logp()
        logp[FREE] = x
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
    return residual


def run_fit(sim, x0, nfev, tag=""):
    lo = np.log(np.maximum(PBOUNDS[FREE, 0], 1e-12))
    hi = np.log(PBOUNDS[FREE, 1])
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    out = least_squares(make_residual(sim), x0, bounds=(lo, hi), method="trf",
                        x_scale="jac", max_nfev=nfev,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    logp = base_logp()
    logp[FREE] = out.x
    p2 = np.exp(logp)[None, :].repeat(sim.n, axis=0)
    m = mape_from_preds(sim.obs_preds(p2), sim)
    print(f"{tag} cost {out.cost:.3f} nfev {out.nfev} MAPE {np.round(m,2)} mean {m.mean():.3f}", flush=True)
    return out.cost, logp, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=24)
    ap.add_argument("--nfev", type=int, default=250)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--subset", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "params_ms.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    B, tpl, df = load_batches()
    obs_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    rng = np.random.default_rng(args.seed)
    sub = list(rng.choice(obs_ids, size=min(args.subset, len(obs_ids)), replace=False))
    sim_sub = SimSet(B, sub, dt=args.dt)
    lo = np.log(np.maximum(PBOUNDS[FREE, 0], 1e-12))
    hi = np.log(PBOUNDS[FREE, 1])

    ckpt = args.out + ".ckpt"
    best = None
    if os.path.exists(ckpt):
        d = json.load(open(ckpt))
        best = (d["cost"], np.array(d["logp"]))
        print(f"resumed best {best[0]:.3f}", flush=True)
    t0 = time.time()
    # staged-ish start
    x0 = np.log(np.maximum(P0[FREE], 1e-9))
    c, logp, m = run_fit(sim_sub, x0, args.nfev, tag="start0")
    if best is None or c < best[0]:
        best = (c, logp)
        json.dump({"cost": best[0], "logp": best[1].tolist()}, open(ckpt, "w"))
    for s in range(args.starts):
        x0 = rng.uniform(lo, hi)
        c, logp, m = run_fit(sim_sub, x0, args.nfev, tag=f"start{s+1}")
        if c < best[0]:
            best = (c, logp)
            json.dump({"cost": best[0], "logp": best[1].tolist()}, open(ckpt, "w"))
            print(f"  new best {c:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"search done ({time.time()-t0:.0f}s), best cost {best[0]:.3f}")

    # polish on all observed batches at finer dt
    sim_all = SimSet(B, obs_ids, dt=min(args.dt, 0.005))
    c, logp, m = run_fit(sim_all, best[1][FREE], max(400, args.nfev), tag="polish-all")
    if c < best[0]:
        best = (c, logp)
    # final polish finer
    sim_fin = SimSet(B, obs_ids, dt=0.0025)
    c, logp, m = run_fit(sim_fin, best[1][FREE], 400, tag="polish-fine")
    if c < best[0]:
        best = (c, logp)

    logp = best[1]
    p2 = np.exp(logp)[None, :].repeat(sim_fin.n, axis=0)
    m = mape_from_preds(sim_fin.obs_preds(p2), sim_fin)
    params = {PNAMES[i]: float(np.exp(logp[i])) for i in range(21)}
    print("FINAL in-sample MAPE", np.round(m, 3), "mean", round(float(m.mean()), 3))
    print("params:", json.dumps({k: round(v, 5) for k, v in params.items()}))
    json.dump({"logp": logp.tolist(), "params": params, "mape": m.tolist(),
               "mean": float(m.mean())}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

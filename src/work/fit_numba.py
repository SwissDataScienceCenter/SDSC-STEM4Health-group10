"""Fit global ODE params (numba sim + scipy least_squares), batch-wise CV.

Usage: python fit_numba.py [--nfev 400] [--folds 6] [--dt 0.0025] [--out params.json]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import (SimSet, load_batches, PNAMES, PBOUNDS, P0, NP, DT,
                       mape_from_preds)


def logp_to_params2d(logp, n):
    return np.exp(logp)[None, :].repeat(n, axis=0)


def make_residual(sim):
    obs_y = sim.obs_y
    obs_n = sim.obs_n
    obs_idx = sim.obs_idx

    def residual(logp):
        p2 = logp_to_params2d(logp, sim.n)
        M_all = sim.simulate(p2)
        res = []
        for j in range(sim.n):
            m = obs_n[j]
            if m <= 1:
                continue
            idx = obs_idx[j, 1:m]  # skip t=0
            Mj = M_all[idx, j, :]
            Vj = sim.V[idx, j]
            pred = Mj / Vj[:, None]
            yo = obs_y[j, 1:m]
            r = (pred - yo) / (np.abs(yo) + 1e-9)
            res.append(r.ravel())
        return np.concatenate(res)
    return residual


def fit_global(sim, nfev=400, x0=None, verbose=True):
    lo = np.log(np.maximum(PBOUNDS[:, 0], 1e-9))
    hi = np.log(PBOUNDS[:, 1])
    if x0 is None:
        x0 = np.log(np.maximum(P0, 1e-6))
    res_fn = make_residual(sim)
    t0 = time.time()
    out = least_squares(res_fn, x0, bounds=(lo, hi), method="trf",
                        x_scale="jac", max_nfev=nfev, verbose=0)
    if verbose:
        print(f"  fit done: cost {out.cost:.4f} nfev {out.nfev} ({time.time()-t0:.0f}s) "
              f"status {out.status}")
    return out.x


def eval_sim(sim, logp, skip_first=True):
    p2 = logp_to_params2d(logp, sim.n)
    preds = sim.obs_preds(p2)
    m = mape_from_preds(preds, sim, skip_first=skip_first)
    return m, float(m.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfev", type=int, default=400)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "params_global.json"))
    ap.add_argument("--cv-only", action="store_true")
    args = ap.parse_args()

    B, tpl, df = load_batches()
    obs_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    hid_ids = sorted([b for b in B if B[b]["hidden"]], key=lambda s: int(s[5:]))
    print(f"observed {len(obs_ids)}, hidden {len(hid_ids)}, dt={args.dt}")

    sim = SimSet(B, obs_ids, dt=args.dt)
    t0 = time.time()
    r = make_residual(sim)(np.log(np.maximum(P0, 1e-6)))
    print(f"one residual eval: {time.time()-t0:.3f}s, n_res={len(r)}")

    print("fitting global params on all observed batches...")
    logp = fit_global(sim, nfev=args.nfev)
    m, mean = eval_sim(sim, logp)
    print("in-sample MAPE (13 pts):", np.round(m, 3), "mean", round(mean, 3))
    params = {n: float(np.exp(logp[i])) for i, n in enumerate(PNAMES)}
    print("params:", json.dumps({k: round(v, 5) for k, v in params.items()}))

    if not args.cv_only:
        json.dump({"logp": logp.tolist(), "params": params, "in_sample": m.tolist(),
                   "mean": mean, "dt": args.dt}, open(args.out, "w"), indent=1)
        print("wrote", args.out)

    if args.folds > 0:
        print(f"\n{args.folds}-fold batch CV (predict from t=0 only)...")
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(obs_ids))
        folds = np.array_split(perm, args.folds)
        cvs = []
        for fi, fold in enumerate(folds):
            te = [obs_ids[i] for i in fold]
            tr = [obs_ids[i] for i in range(len(obs_ids)) if i not in set(fold.tolist())]
            sim_tr = SimSet(B, tr, dt=args.dt)
            logp_f = fit_global(sim_tr, nfev=max(150, args.nfev // 2), verbose=False)
            sim_te = SimSet(B, te, dt=args.dt)
            m, mean = eval_sim(sim_te, logp_f)
            cvs.append(m)
            print(f"fold {fi}: n={len(te)} MAPE {np.round(m,3)} mean {mean:.3f}", flush=True)
        cv = np.mean(cvs, axis=0)
        print("CV MAPE:", np.round(cv, 3), "mean", round(float(cv.mean()), 3))
        if not args.cv_only:
            d = json.load(open(args.out))
            d["cv_mape"] = cv.tolist()
            d["cv_mean"] = float(cv.mean())
            json.dump(d, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

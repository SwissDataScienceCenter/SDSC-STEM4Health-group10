"""Fit global ODE params on observed batches; validate with batch-wise CV.

Usage: python fit_global.py [--iters 400] [--folds 6] [--dt 0.0025] [--out params.json]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from ode_torch import (Sim, load_batches, PNAMES, PBOUNDS, P0, init_log,
                       log_bounds, mape_per_target, rel_residuals, find_data_dir)

torch.set_num_threads(16)


def fit(sim, iters=400, lr=0.05, seed=0, loss_pow=2.0, verbose=True, logp_init=None):
    lo, hi = log_bounds()
    lo_t = torch.as_tensor(lo, dtype=torch.float64)
    hi_t = torch.as_tensor(hi, dtype=torch.float64)
    if logp_init is None:
        logp = torch.tensor(init_log(), dtype=torch.float64, requires_grad=True)
    else:
        logp = torch.tensor(logp_init, dtype=torch.float64, requires_grad=True)
    obs_y_t = [torch.as_tensor(y, dtype=torch.float64) for y in sim.obs_y]
    opt = torch.optim.Adam([logp], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.05)
    best = (1e18, None)
    t0 = time.time()
    for it in range(iters):
        opt.zero_grad()
        preds = sim.obs_preds(logp)
        loss = 0.0
        for pr, yt in zip(preds, obs_y_t):
            r = (pr - yt) / (yt.abs() + 1e-8)
            loss = loss + (r ** loss_pow).mean()
        loss.backward()
        with torch.no_grad():
            logp.clamp_(lo_t + 1e-6, hi_t - 1e-6)
        opt.step()
        sched.step()
        lv = float(loss)
        if lv < best[0]:
            best = (lv, logp.detach().clone())
        if verbose and (it % 50 == 0 or it == iters - 1):
            print(f"  it {it:4d} loss {lv:.5f} best {best[0]:.5f} ({time.time()-t0:.0f}s)", flush=True)
    return best[1]


def eval_mape(sim, logp, skip_first_obs=True):
    preds = sim.obs_preds(logp)
    obs = []
    prs = []
    for j, pr in enumerate(preds):
        yo = sim.obs_y[j]
        if skip_first_obs and len(yo) > 1:
            prs.append(pr[1:])
            obs.append(yo[1:])
        else:
            prs.append(pr)
            obs.append(yo)
    m = mape_per_target(prs, obs)
    mean = float(np.mean([m[t] for t in m]))
    return m, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.0025)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--loss-pow", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "params_global.json"))
    ap.add_argument("--cv-only", action="store_true")
    args = ap.parse_args()

    B, tpl, df = load_batches()
    obs_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    hid_ids = sorted([b for b in B if B[b]["hidden"]], key=lambda s: int(s[5:]))
    print(f"observed batches: {len(obs_ids)}, hidden: {len(hid_ids)}")

    # full fit
    sim = Sim(B, obs_ids, dt=args.dt)
    print("fitting global params on all observed batches...")
    logp = fit(sim, iters=args.iters, lr=args.lr, loss_pow=args.loss_pow)
    m, mean = eval_mape(sim, logp)
    print("in-sample MAPE (13 pts):", {k: round(v, 3) for k, v in m.items()}, "mean", round(mean, 3))
    params = {n: float(np.exp(logp.detach().numpy()[i])) for i, n in enumerate(PNAMES)}
    print("params:", json.dumps({k: round(v, 5) for k, v in params.items()}, indent=1))
    if not args.cv_only:
        with open(args.out, "w") as f:
            json.dump({"logp": logp.detach().numpy().tolist(), "params": params,
                       "in_sample_mape": m, "mean": mean, "dt": args.dt}, f, indent=1)
        print("wrote", args.out)

    # CV: fold observed batches, fit on rest, evaluate on held-out (sim from t=0)
    print(f"\n{args.folds}-fold batch CV (leaderboard-like: predict from t=0 only)...")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs_ids))
    folds = np.array_split(perm, args.folds)
    all_m = {t: [] for t in (0, 1, 2, 3)}
    for fi, fold in enumerate(folds):
        te = [obs_ids[i] for i in fold]
        tr = [obs_ids[i] for i in range(len(obs_ids)) if i not in set(fold.tolist())]
        sim_tr = Sim(B, tr, dt=args.dt)
        logp_f = fit(sim_tr, iters=max(150, args.iters // 2), lr=args.lr, loss_pow=args.loss_pow, verbose=False)
        sim_te = Sim(B, te, dt=args.dt)
        m, mean = eval_mape(sim_te, logp_f)
        print(f"fold {fi}: holdout {len(te)} batches MAPE " +
              str({k: round(v, 3) for k, v in m.items()}) + f" mean {mean:.3f}", flush=True)
        for t in all_m:
            all_m[t].append(m[t])
    cv = {t: float(np.mean(all_m[t])) for t in all_m}
    print("CV MAPE:", {k: round(v, 3) for k, v in cv.items()},
          "mean", round(float(np.mean(list(cv.values()))), 3))
    if not args.cv_only:
        with open(args.out, "a") as f:
            pass
        d = json.load(open(args.out))
        d["cv_mape"] = cv
        d["cv_mean"] = float(np.mean(list(cv.values())))
        json.dump(d, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

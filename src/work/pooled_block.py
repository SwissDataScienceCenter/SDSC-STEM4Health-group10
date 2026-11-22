"""Pooled coupled ODE fit over all 49 observed batches, block coordinate descent
with multi-start, checkpointed. Usage:
python pooled_block.py --rounds 1 --starts 4 --nfev 150 --dt 0.01 --out pooled.json
"""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import SimSet, load_batches, PNAMES, PBOUNDS, P0, mape_from_preds
from coupled_perbatch import BlockFit, GIDX, MIDX, base_logp
from scipy.optimize import least_squares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--nfev", type=int, default=150)
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "pooled.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    B, tpl, df = load_batches()
    obs_ids = sorted([b for b in B if not B[b]["hidden"]], key=lambda s: int(s[5:]))
    sim = SimSet(B, obs_ids, dt=args.dt)
    bf = BlockFit(sim)
    rng = np.random.default_rng(args.seed)
    ckpt = args.out + ".ckpt"
    if os.path.exists(ckpt):
        d = json.load(open(ckpt))
        logp = np.array(d["logp"]); print("resumed cost", d["cost"], flush=True)
    elif os.path.exists(args.out):
        d = json.load(open(args.out))
        logp = np.array(d["logp"])
    else:
        logp = base_logp()
    t0 = time.time()
    for r in range(args.rounds):
        logp, cg = bf.fit_block(logp, GIDX, (0, 1), args.starts, args.nfev, rng)
        print(f"round {r} growth cost {cg:.3f} ({time.time()-t0:.0f}s)", flush=True)
        logp, cm = bf.fit_block(logp, MIDX, (2, 3), args.starts, args.nfev, rng)
        print(f"round {r} metab cost {cm:.3f} ({time.time()-t0:.0f}s)", flush=True)
        m = mape_from_preds(sim.obs_preds(np.exp(logp)[None, :].repeat(sim.n, 0)), sim)
        print(f"round {r} MAPE {np.round(m,2)} mean {m.mean():.2f}", flush=True)
        json.dump({"logp": logp.tolist(), "cost": float(cm)}, open(ckpt, "w"))
    if args.polish:
        logp, ca = bf.fit_all(logp, starts=args.starts, nfev=args.nfev, rng=rng)
        print(f"polish cost {ca:.3f}", flush=True)
    m = mape_from_preds(sim.obs_preds(np.exp(logp)[None, :].repeat(sim.n, 0)), sim)
    print("FINAL MAPE", np.round(m, 3), "mean", round(float(m.mean()), 3))
    print("params:", {PNAMES[i]: round(float(np.exp(logp[i])), 5) for i in range(21)})
    json.dump({"logp": logp.tolist(), "params": {PNAMES[i]: float(np.exp(logp[i])) for i in range(21)},
               "mape": m.tolist(), "mean": float(m.mean())}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

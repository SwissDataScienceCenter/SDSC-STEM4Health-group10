"""Re-tuning of the constrained controller update (kalman3 full, LB 7.300) now that the constraints are inside.
Stage A: observation days. Stage B: shrink, noise scale, tau_mid, late-rule quantile. Choice on seeds 0-2.
Usage: python src/experiments/controller_kalman4.py"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import controller_kalman as K, controller_kalman3 as K3, control_rules as C

_LateOrig = C.LateDoseRule


def score(days, shrink=0.4, sR=1.5, tau_mid=0.1, q=0.2):
    K3.KS, K3.SHRINK, K3.SR = days, shrink, sR
    C.LateDoseRule = lambda q_=q: _LateOrig(q_)            # used inside K3.run and K.late_rule
    b = [K.mape(K3.run(s, tau_late=0.03, tau_mid=tau_mid)) for s in range(6)]
    C.LateDoseRule = _LateOrig
    return np.mean(b[:3]), np.mean(b[3:])


def main():
    base = [1, 2, 4, 5, 6, 7, 8, 11]
    ref = score(base)
    print(f"reference kalman3 full (LB 7.300): {ref[0]:.3f} / {ref[1]:.3f}", flush=True)
    days = {"+3": sorted(base + [3]), "+9": sorted(base + [9]), "+10": sorted(base + [10]), "+12": base + [12], "+13": base + [13],
            "+12,13": base + [12, 13], "-8": [1, 2, 4, 5, 6, 7, 11], "+9,10": sorted(base + [9, 10]), "all 1-13": list(range(1, 14))}
    res = {"base": ref}
    for n, d in days.items():
        res[n] = score(d); print(f"A {n:9s}: {res[n][0]:.3f} / {res[n][1]:.3f}", flush=True)
    best = min(res, key=lambda n: res[n][0]); d = base if best == "base" else days[best]
    print(f"-> days: {best} {d}", flush=True)
    res2 = {}
    for shrink in (0.3, 0.4, 0.6):
        for sR in (1.25, 1.5, 2.0):
            res2[(shrink, sR, 0.1, 0.2)] = score(d, shrink, sR); print(f"B shrink={shrink} R x{sR}: {res2[(shrink, sR, 0.1, 0.2)][0]:.3f} / {res2[(shrink, sR, 0.1, 0.2)][1]:.3f}", flush=True)
    bs = min(res2, key=lambda c: res2[c][0])
    for tau_mid in (0.05, 0.15):
        c = (bs[0], bs[1], tau_mid, 0.2); res2[c] = score(d, bs[0], bs[1], tau_mid); print(f"B tau_mid={tau_mid}: {res2[c][0]:.3f} / {res2[c][1]:.3f}", flush=True)
    bt = min(res2, key=lambda c: res2[c][0])
    for q in (0.1, 0.3):
        c = (bt[0], bt[1], bt[2], q); res2[c] = score(d, bt[0], bt[1], bt[2], q); print(f"B late q={q}: {res2[c][0]:.3f} / {res2[c][1]:.3f}", flush=True)
    bb = min(res2, key=lambda c: res2[c][0])
    print(f"\nBEST (seeds 0-2): days {d}, shrink {bb[0]}, R x{bb[1]}, tau_mid {bb[2]}, late q {bb[3]}: {res2[bb][0]:.3f} / {res2[bb][1]:.3f}"
          f"  vs reference {ref[0]:.3f} / {ref[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()

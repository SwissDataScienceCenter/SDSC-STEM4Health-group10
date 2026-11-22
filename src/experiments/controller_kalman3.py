"""Controller update with the control-rule constraints as pseudo-observations (propagated through the prior
correlations), on top of the tuned update (days 1,2,4-8,11, shrink 0.4, noise x1.5). Choice on seeds 0-2."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import controller_kalman as K, controller_kalman2 as K2, control_rules as C, evaluation as E
K.current_best = K2.blend_base_substrate
KS, SHRINK, SR = [1, 2, 4, 5, 6, 7, 8, 11], 0.4, 1.5
BIG = C.late_dose_classes(K.raw)
TRIG = C.trigger_samples(K.raw)


def obs_rows(b, laws, s0):
    H, z, Rv = [], [], []
    for k, (lr, sig, cols) in laws.items():
        c = dict(zip(cols, lr.coef_)); h = np.zeros(13); h[k - 1] = 1.0
        rhs = lr.intercept_ + c.get("d", 0) * K.DOSE.loc[b, k] + (c.get("dp", 0) * K.DOSE.loc[b, k - 1] if k > 1 else 0) + c.get("fa", 0) * K.FA.loc[b, k]
        if k == 1: rhs += c["Sp"] * s0
        else: h[k - 2] = -c["Sp"]
        H.append(h); z.append(rhs); Rv.append((SR * sig) ** 2)
    return H, z, Rv


def solve(mu, Sig, H, z, Rv):
    H, z = np.array(H), np.array(z)
    return mu + Sig @ H.T @ np.linalg.inv(H @ Sig @ H.T + np.diag(Rv)) @ (z - H @ mu)


def run(s, tau_late=0.03, tau_mid=None, mid_equal=True, iters=3):
    P = K.current_best(s); Pm = P.unstack()[K.J]
    rel = (Pm - K.S_all.loc[Pm.index, K.J]) / K.S_all.loc[Pm.index, K.J]
    new = Pm.copy()
    for fo in E.make_folds(K.TRAIN, 7, s):
        tr = [b for b in K.TRAIN if b not in fo]
        laws = K.fit_controller(tr, KS); sd, Cm = K.prior_cov(rel.loc[tr], SHRINK)
        late = C.LateDoseRule().fit(K.raw, tr); mid = C.SubstrateControlRule().fit(K.raw, tr)
        for b in fo:
            mu = Pm.loc[b].to_numpy(); s0 = K.S_all.loc[b, 0]; Sig = np.outer(sd * mu, sd * mu) * Cm
            H, z, Rv = obs_rows(b, laws, s0)
            fixed = []                                   # mid-run bolus: S(j0) ~ c_trig, S(j0+1) ~ c_after (equalities)
            if tau_mid is not None and mid_equal and b in TRIG.index:
                j0 = int(TRIG[b]); fixed = [(j0, mid.c_trig), (j0 + 1, mid.c_after)]
            Hx, zx, Rx = list(H), list(z), list(Rv)
            for k, val in fixed:
                h = np.zeros(13); h[k - 1] = 1.0; Hx.append(h); zx.append(val); Rx.append(tau_mid ** 2)
            post = solve(mu, Sig, Hx, zx, Rx)
            for _ in range(iters):                       # violated inequalities -> pseudo-observations at the bound
                extra = []
                for k in (11, 12):
                    big = bool(BIG.loc[(b, k)])
                    if big and post[k - 1] > late.cap: extra.append((k, late.cap, tau_late))
                    if (not big) and post[k - 1] < late.floor: extra.append((k, late.floor, tau_late))
                if tau_mid is not None:
                    j0 = TRIG.get(b, np.nan)
                    for k in C.WINDOW:
                        if (np.isnan(j0) or k != j0) and not (not np.isnan(j0) and k == j0 + 1) and post[k - 1] < mid.floor:
                            extra.append((k, mid.floor, tau_mid))
                if not extra: break
                Hy, zy, Ry = list(Hx), list(zx), list(Rx)
                for k, val, t in extra:
                    h = np.zeros(13); h[k - 1] = 1.0; Hy.append(h); zy.append(val); Ry.append(t ** 2)
                post = solve(mu, Sig, Hy, zy, Ry)
            new.loc[b] = np.maximum(post, 0.05)
    new = new.stack(); new.index.names = ["batch_id", "j"]
    return K.late_rule(new.loc[P.index], s)


if __name__ == "__main__":
    for name, kw in [("late tau 0.03", dict(tau_late=0.03)), ("late tau 0.01", dict(tau_late=0.01)),
                     ("late 0.03 + mid-run 0.1", dict(tau_late=0.03, tau_mid=0.1)), ("late 0.03 + mid-run 0.2", dict(tau_late=0.03, tau_mid=0.2)),
                     ("late 0.03 + mid-run floor only 0.1", dict(tau_late=0.03, tau_mid=0.1, mid_equal=False))]:
        b = [K.mape(run(s, **kw)) for s in range(6)]
        print(f"{name:36s}: {np.mean(b[:3]):.3f} / {np.mean(b[3:]):.3f}", flush=True)

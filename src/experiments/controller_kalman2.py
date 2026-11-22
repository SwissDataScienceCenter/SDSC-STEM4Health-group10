"""Stage-wise tuning of the Feed B controller update (prior = blend_base substrate), choice on seeds 0-2.
Adds a log-space variant (laws and prior on log S). Usage: python src/experiments/controller_kalman2.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import controller_kalman as K
import evaluation as E


def blend_base_substrate(s):
    ou = pd.read_csv(K.OUT / f"ours_oof_s{s}.csv").set_index(["batch_id", "j"])
    th = pd.read_csv(K.OUT / f"their_oof_s{s}.csv").set_index(["batch_id", "j"]).loc[ou.index]
    return np.exp(0.5 * np.log(ou.substrate) + 0.5 * np.log(th.substrate))


K.current_best = blend_base_substrate
PRIORS = {s: blend_base_substrate(s) for s in range(6)}


def fit_log(tr, ks):
    laws = {}
    for k in ks:
        X = pd.DataFrame({"Sp": np.log(K.S_all.loc[tr, k - 1]), "d": K.DOSE.loc[tr, k]})
        if k > 1:
            X["dp"] = K.DOSE.loc[tr, k - 1]
        if (K.FA.loc[tr, k] > 0).sum() >= 3:
            X["fa"] = K.FA.loc[tr, k]
        X = X.loc[:, X.std() > 0]
        y = np.log(K.S_all.loc[tr, k])
        lr = LinearRegression().fit(X, y)
        laws[k] = (lr, float(np.std(y - lr.predict(X), ddof=X.shape[1] + 1)), list(X.columns))
    return laws


def run_log(s, ks, shrink, scale_R):
    P = PRIORS[s]; Pm = P.unstack()[K.J]
    lres = np.log(Pm) - np.log(K.S_all.loc[Pm.index, K.J])
    new = Pm.copy()
    for fo in E.make_folds(K.TRAIN, 7, s):
        tr = [b for b in K.TRAIN if b not in fo]
        laws = fit_log(tr, ks)
        sd = lres.loc[tr].std().to_numpy()
        Cm = (1 - shrink) * lres.loc[tr].corr().to_numpy() + shrink * np.eye(13)
        Sig = np.outer(sd, sd) * Cm
        for b in fo:
            mu = np.log(Pm.loc[b].to_numpy())
            H, z, Rv = [], [], []
            for k, (lr, sig, cols) in laws.items():
                c = dict(zip(cols, lr.coef_)); h = np.zeros(13); h[k - 1] = 1.0
                rhs = lr.intercept_ + c.get("d", 0) * K.DOSE.loc[b, k] + (c.get("dp", 0) * K.DOSE.loc[b, k - 1] if k > 1 else 0) + c.get("fa", 0) * K.FA.loc[b, k]
                if k == 1:
                    rhs += c["Sp"] * np.log(K.S_all.loc[b, 0])
                else:
                    h[k - 2] = -c["Sp"]
                H.append(h); z.append(rhs); Rv.append((scale_R * sig) ** 2)
            H, z = np.array(H), np.array(z)
            G = Sig @ H.T @ np.linalg.inv(H @ Sig @ H.T + np.diag(Rv))
            new.loc[b] = np.exp(mu + G @ (z - H @ mu))
    new = new.stack(); new.index.names = ["batch_id", "j"]
    return K.late_rule(new.loc[P.index], s)


def score(ks, shrink, scale_R, log=False):
    new = [K.mape(run_log(s, ks, shrink, scale_R) if log else K.run(s, ks, shrink, scale_R)[1]) for s in range(6)]
    return np.mean(new[:3]), np.mean(new[3:])


def main():
    b6 = [1, 2, 4, 5, 6, 7]
    ref = score(b6, 0.6, 1.5)
    print(f"reference (validated, LB 7.438): days {b6} shrink 0.6 R x1.5: {ref[0]:.3f} / {ref[1]:.3f}", flush=True)
    days = {"b6+11": b6 + [11], "b6+3": sorted(b6 + [3]), "b6+3+11": sorted(b6 + [3, 11]), "b6+8": b6 + [8], "b6+8+11": b6 + [8, 11],
            "b6+11,12,13": b6 + [11, 12, 13], "b6+9,11": b6 + [9, 11], "b6+10,11": b6 + [10, 11]}
    res = {}
    for n, ks in days.items():
        res[n] = score(ks, 0.6, 1.5)
        print(f"stage 1  {n:12s}: {res[n][0]:.3f} / {res[n][1]:.3f}", flush=True)
    best = min(res, key=lambda n: res[n][0]); ks = days[best]
    print(f"-> best day set on seeds 0-2: {best}", flush=True)
    res2 = {}
    for shrink in (0.4, 0.6, 0.8):
        for sR in (1.25, 1.5, 2.0):
            for log in (False, True):
                res2[(shrink, sR, log)] = score(ks, shrink, sR, log)
                print(f"stage 2  shrink={shrink} R x{sR} log={log}: {res2[(shrink, sR, log)][0]:.3f} / {res2[(shrink, sR, log)][1]:.3f}", flush=True)
    b2 = min(res2, key=lambda c: res2[c][0])
    print(f"\nBEST (seeds 0-2): days {best} {ks}, shrink {b2[0]}, R x{b2[1]}, log {b2[2]}: {res2[b2][0]:.3f} / {res2[b2][1]:.3f}"
          f"  vs reference {ref[0]:.3f} / {ref[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()

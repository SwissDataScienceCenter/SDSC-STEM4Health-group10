"""Hybrid serial model: ridge specific rates + bolus mass balance, simulated step by step."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import common as X
D = X.D
CB, CS = 50.0, 250.0
RATES = ["mu", "kd", "qS", "qP"]
FNAMES = ["lXv", "lXd", "S", "P", "lS", "lP", "t", "j", "FB", "FS", "cFB", "cFS"]

_B = {}
def batches():
    if not _B:
        raw = X.load()[0]
        m = D.measurements(raw, include_to_predict=True)
        f = D.feeds_long(raw)
        for b, g in m.groupby("batch_id"):
            g = g.sort_values("elapsed_time")
            t = g.elapsed_time.to_numpy(); n = len(t)
            fb = np.zeros(n - 1); fs = np.zeros(n - 1)
            ff = f[f.batch_id == b]
            for k in range(n - 1):
                w = ff[(ff.elapsed_time > t[k]) & (ff.elapsed_time <= t[k + 1])]
                fb[k] = w.loc[w.feed == "feed_b", "amount"].sum()
                fs[k] = w.loc[w.feed == "feed_substrate", "amount"].sum()
            _B[b] = dict(t=t, V=g.volume.to_numpy(), fb=fb, fs=fs, cfb=np.r_[0, np.cumsum(fb)], cfs=np.r_[0, np.cumsum(fs)],
                         st=g[["vcd", "dcd", "substrate", "byproduct"]].to_numpy())
    return _B

def feat(b, k, s):
    Xv, Xd, S, P = np.maximum(s, 1e-3)
    return [np.log(Xv), np.log(Xd), S, P, np.log(S), np.log(P), b["t"][k], k, b["fb"][k], b["fs"][k], b["cfb"][k], b["cfs"][k]]

def true_rates(b, k, s0, V0):
    """rates linking current (possibly simulated) amounts s0,V0 to TRUE next state."""
    dt = b["t"][k + 1] - b["t"][k]; V1 = b["V"][k + 1]; s1 = b["st"][k + 1]
    a0 = s0 * V0; a1 = s1 * V1
    mu = np.log(a1[0] / a0[0]) / dt
    A = (a0[0] + a1[0]) / 2
    kd = (a1[1] - a0[1]) / (dt * A)
    qS = (a0[2] + CB * b["fb"][k] + CS * b["fs"][k] - a1[2]) / (dt * A)
    qP = (a1[3] - a0[3]) / (dt * A)
    return np.array([mu, kd, qS, qP])

def step(b, k, s0, r):
    dt = b["t"][k + 1] - b["t"][k]; V0 = b["V"][k]; V1 = b["V"][k + 1]
    a0 = s0 * V0
    xv1 = a0[0] * np.exp(r[0] * dt); A = (a0[0] + xv1) / 2
    a1 = np.array([xv1, a0[1] + r[1] * dt * A,
                   a0[2] + CB * b["fb"][k] + CS * b["fs"][k] - r[2] * dt * A, a0[3] + r[3] * dt * A])
    return np.maximum(a1 / V1, [1e-3, 1e-3, 1e-2, 1e-2])

class Model:
    def __init__(self): self.m = None
    def fit(self, F, Y):
        self.lo, self.hi = np.percentile(Y, 1, 0), np.percentile(Y, 99, 0)
        self.m = [make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(F, Y[:, i]) for i in range(4)]
        return self
    def rates(self, f):
        f = np.array(f)[None]
        return np.clip(np.array([m.predict(f)[0] for m in self.m]), self.lo, self.hi)

def dataset(ids, B):
    F, Y = [], []
    for i in ids:
        b = B[i]
        for k in range(12):
            s = b["st"][k]; F.append(feat(b, k, s)); Y.append(true_rates(b, k, s, b["V"][k]))
    return np.array(F), np.array(Y)

def simulate(model, b, s_start):
    s = np.maximum(s_start, [1e-3, 1e-3, 1e-2, 1e-2]); out = [s]
    for k in range(12):
        s = step(b, k, s, model.rates(feat(b, k, s))); out.append(s)
    return np.array(out)

def make_fp(dagger=False, blend=0.0):
    def fp(train_ids, ids):
        B = batches(); F, Y = dataset(train_ids, B); model = Model().fit(F, Y)
        if dagger:
            F2, Y2 = [F], [Y]
            for i in train_ids:
                b = B[i]; s = b["st"][0]
                for k in range(12):
                    r = model.rates(feat(b, k, s)); F2.append([feat(b, k, s)]); Y2.append([true_rates(b, k, s, b["V"][k])])
                    s = step(b, k, s, r)
            model = Model().fit(np.vstack([np.atleast_2d(a) for a in F2]), np.vstack([np.atleast_2d(a) for a in Y2]))
        rows = []
        for i in ids:
            b = B[i]; sim = simulate(model, b, b["st"][0])
            for j in range(1, 13 + 1):
                if j >= len(b["t"]): break
                v, d, S, P = sim[j]
                rows.append(dict(batch_id=i, j=j, elapsed_time=b["t"][j], vcd=v, tcd=v + d, substrate=S, byproduct=P))
        p = pd.DataFrame(rows)
        key = X.load()[1][["batch_id", "j", "elapsed_time"]]
        p = key[key.batch_id.isin(ids)].merge(p.drop(columns="elapsed_time"), on=["batch_id", "j"], how="left")
        p = X._sorted(p)
        if blend: p = X.geo_blend(p, X.baseline_scaled(train_ids, ids), blend)
        return p
    return fp

if __name__ == "__main__":
    t0 = time.time()
    raw, table, train, test = X.load()
    oof = X.lobo(make_fp()); X.report("serial_raw", oof); print(time.time() - t0, flush=True)
    sb = X.report("serial_blend", X.geo_blend(oof, X.lobo(X.baseline_scaled), 0.2))
    print(time.time() - t0, flush=True)
    oofd = X.lobo(make_fp(dagger=True)); X.report("serial_noisy", oofd); print(time.time() - t0, flush=True)
    # coefficients
    B = batches(); F, Y = dataset(train, B); m = Model().fit(F, Y)
    for r, mm in zip(RATES, m.m):
        c = mm[-1].coef_; o = np.argsort(-abs(c))[:3]
        print(r, [(FNAMES[i], round(c[i], 3)) for i in o])

"""
Decode the Bioreactor - Submission A: mechanistic fed-batch model (M4), self-contained.

Run:   python main.py   (uses the Renku data paths below; or pass the two CSV paths as arguments)
Output: submission_A_mechanistic.csv (same rows/columns as the template) + fitted_parameters.json

Model (ODEs, concentrations in the tank):
  Xv viable cells, Xd dead cells (TCD = Xv + Xd), S substrate, P byproduct,
  A unmeasured inhibitor (ammonia-like) made by live cells: dA/dt = Xv
  growth     mu = mu_max * (1 - exp(-t/tlag)) * KA^nA / (KA^nA + A^nA)
  death      kd = kd0 + kd1*Kd^2/(Kd^2+S^2) + kdA*A^nA/(KdA^nA+A^nA)
  cells      dXv/dt = (mu - kd) Xv            dXd/dt = kd Xv - klys Xd
  overflow   O = q_ov * S^n/(Kov^n+S^n) * Kx/(Kx+Xv)
  substrate  dS/dt = -(m + mu/Yxs + O) Xv
  byproduct  dP/dt = (Yp*O - q_c*P/(Kc+P)) Xv
  feeds      bolus at recorded times: everything diluted by V_before/V_after,
             substrate += (c_A*feed_substrate + c_B*feed_b)/V_after
Parameters are shared by all batches and fitted on the complete batches by minimising
relative error (matches the leaderboard MAPE). Each test batch is forecast from its day-0 values.
"""
import sys, json, time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
OBS = ["vcd", "tcd", "substrate", "byproduct"]
DT, T_END = 0.02, 13.1

PARAMS = {  # name: (start, lower, upper)
    "mu_max": (0.9, 0.3, 2.5), "KA": (50.0, 5, 500), "nA": (2.0, 0.5, 6),
    "kdA": (0.03, 1e-4, 0.5), "KdA": (200.0, 20, 2000), "tlag": (0.3, 0.01, 2),
    "kd0": (0.007, 1e-4, 0.1), "kd1": (0.03, 1e-4, 0.3), "Kd": (1.0, 0.05, 3),
    "klys": (0.03, 1e-4, 0.5), "m": (0.105, 0.01, 0.3), "Yxs": (30.0, 0.5, 2000),
    "q_ov": (0.65, 1e-3, 5), "Kov": (3.0, 0.5, 10), "n_ov": (5.0, 1.0, 10),
    "Kx": (3.0, 0.1, 50), "Yp": (1.0, 0.05, 3), "q_c": (0.02, 1e-5, 0.3),
    "Kc": (0.5, 0.01, 5), "c_B": (52.0, 20, 120), "c_A": (250.0, 80, 600),
}
NAMES = list(PARAMS)


def prepare(d, batches):
    n_steps = int(round(T_END / DT)); grid = np.arange(n_steps + 1) * DT; B = len(batches)
    F = {f: np.zeros((n_steps, B)) for f in FEEDS}; V = np.zeros((n_steps + 1, B)); x0 = np.zeros((5, B))
    samples = []
    for j, b in enumerate(batches):
        g = d[d.batch_id == b].sort_values("elapsed_time")
        for f in FEEDS:
            ev = g[g[f].notna()]
            k = np.minimum((ev.elapsed_time.values / DT).astype(int), n_steps - 1)
            np.add.at(F[f][:, j], k, ev[f].values)
        vm = g[g.volume.notna()]
        cum = lambda t: sum(g.loc[g.elapsed_time <= t, f].sum() for f in FEEDS)
        base = np.array([v - cum(t) for t, v in zip(vm.elapsed_time, vm.volume)])
        V[:, j] = np.interp(grid, vm.elapsed_time.values, base) + np.r_[0, np.cumsum(sum(F[f][:, j] for f in FEEDS))]
        s0 = g[(g.elapsed_time == 0) & g.vcd.notna()].iloc[0]
        x0[:, j] = [s0.vcd, s0.tcd - s0.vcd, s0.substrate, s0.byproduct, 0.0]
        for _, r in g[(g.vcd.notna() | g.to_predict) & (g.elapsed_time > 0)].iterrows():
            k = min(int(np.ceil(r.elapsed_time / DT - 1e-9)), n_steps)
            samples.append((j, r.elapsed_time, k, r[OBS].values.astype(float), b))
    kk = np.array([max(s[2], 1) for s in samples]); jj = np.array([s[0] for s in samples])
    w = np.clip([(s[1] - (k - 1) * DT) / DT for s, k in zip(samples, kk)], 0, 1)
    return dict(F=F, V=V, x0=x0, samples=samples, n_steps=n_steps, kk=kk, jj=jj, w=np.array(w))


def rates(x, p, t):
    Xv, Xd, S, P, A = np.maximum(x, 0)
    An = A ** p["nA"]
    mu = p["mu_max"] * (1 - np.exp(-t / p["tlag"])) * p["KA"] ** p["nA"] / (p["KA"] ** p["nA"] + An)
    kd = p["kd0"] + p["kd1"] * p["Kd"] ** 2 / (p["Kd"] ** 2 + S ** 2) + p["kdA"] * An / (p["KdA"] ** p["nA"] + An)
    Sn = S ** p["n_ov"]
    O = p["q_ov"] * Sn / (p["Kov"] ** p["n_ov"] + Sn) * p["Kx"] / (p["Kx"] + Xv)
    qS = np.where(S > 1e-4, p["m"] + np.maximum(mu, 0) / p["Yxs"] + O, 0)
    qP = p["Yp"] * O - p["q_c"] * P / (p["Kc"] + P)
    return np.array([(mu - kd) * Xv, kd * Xv - p["klys"] * Xd, -qS * Xv, qP * Xv, Xv])


def simulate(p, prep):
    F, V, n = prep["F"], prep["V"], prep["n_steps"]
    x = prep["x0"].copy(); out = np.empty((n + 1, 5, x.shape[1])); post = np.empty_like(out); out[0] = x
    for k in range(n):
        xprev, t = x, k * DT
        k1 = rates(x, p, t); k2 = rates(x + DT / 2 * k1, p, t + DT / 2)
        k3 = rates(x + DT / 2 * k2, p, t + DT / 2); k4 = rates(x + DT * k3, p, t + DT)
        x = np.maximum(x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0)
        post[k] = xprev; out[k + 1] = x                      # out = state just before any feed
        add = sum(F[f][k] for f in FEEDS)
        if add.any():
            x = x * (V[k + 1] - add) / V[k + 1]
            x[2] += (p["c_B"] * F["feed_b"][k] + p["c_A"] * F["feed_substrate"][k]) / V[k + 1]
    post[n] = x
    kk, jj, w = prep["kk"], prep["jj"], prep["w"]
    st = ((1 - w)[:, None] * post[kk - 1, :, jj] + w[:, None] * out[kk, :, jj])[:, :4]
    return np.c_[st[:, 0], st[:, 0] + st[:, 1], st[:, 2], st[:, 3]]   # vcd, tcd, substrate, byproduct


def fit(prep, p, free, use, nfev):
    lo = np.log([PARAMS[n][1] for n in free]); hi = np.log([PARAMS[n][2] for n in free])
    th0 = np.clip(np.log([p[n] for n in free]), lo + 1e-6, hi - 1e-6)
    obs = np.array([s[3] for s in prep["samples"]])

    def res(th):
        q = dict(p, **dict(zip(free, np.exp(th))))
        r = ((simulate(q, prep) - obs) / np.abs(obs))[:, list(use)]
        return np.nan_to_num(r, nan=0.0).ravel()

    sol = least_squares(res, th0, bounds=(lo, hi), loss="soft_l1", f_scale=0.15, max_nfev=nfev, diff_step=1e-3)
    return dict(p, **dict(zip(free, np.exp(sol.x))))


def mape(pred, obs):
    m = np.isfinite(obs)
    return 100 * np.mean(np.abs(pred[m] - obs[m]) / obs[m])


if __name__ == "__main__":
    if len(sys.argv) > 2:
        data_path, tmpl_path = sys.argv[1], sys.argv[2]
    else:
        data_path = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv"
        tmpl_path = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_template.csv"
    d = pd.read_csv(data_path); tmpl = pd.read_csv(tmpl_path)
    n = d.groupby("batch_id").vcd.count(); train = sorted(n[n > 1].index)
    test = list(dict.fromkeys(tmpl.batch_id))
    print(f"{len(train)} training batches, {len(test)} test batches")

    prep = prepare(d, train)
    p = {k: v[0] for k, v in PARAMS.items()}
    stages = [("cells", ["mu_max", "KA", "nA", "kdA", "KdA", "tlag", "kd0", "kd1", "Kd", "klys"], (0, 1), 40),
              ("substrate", ["m", "Yxs", "c_B", "c_A", "q_ov", "Kov", "n_ov", "Kx"], (2, 3), 25),
              ("byproduct", ["Yp", "q_c", "Kc"], (3,), 30),
              ("joint", NAMES, (0, 1, 2, 3), 30)]
    obs = np.array([s[3] for s in prep["samples"]])
    for name, free, use, nf in stages:
        t0 = time.time(); p = fit(prep, p, free, use, nf); pr = simulate(p, prep)
        print(f"stage {name:9s} {time.time()-t0:5.0f}s  in-sample MAPE " +
              " ".join(f"{o}={mape(pr[:, i], obs[:, i]):.2f}" for i, o in enumerate(OBS)))
    json.dump(p, open("fitted_parameters.json", "w"), indent=1)

    prep_t = prepare(d, test); pt = simulate(p, prep_t)
    pred = pd.DataFrame(pt, columns=OBS)
    pred["key"] = [f"{s[4]}|{round(s[1], 6)}" for s in prep_t["samples"]]
    pred = pred.set_index("key")
    out = tmpl.copy(); keys = out.batch_id + "|" + out.elapsed_time.round(6).astype(str)
    assert keys.isin(pred.index).all(), "some template rows were not predicted"
    for o in OBS:
        out[o] = np.maximum(pred.loc[keys, o].values, 1e-3)
    assert list(out.columns) == list(tmpl.columns) and len(out) == len(tmpl) and out[OBS].notna().all().all()
    out.to_csv("submission_A_mechanistic.csv", index=False)
    print(f"wrote submission_A_mechanistic.csv ({len(out)} rows) and fitted_parameters.json")
    
"""
Decode the Bioreactor - final model (iteration v3), one self-contained script.

Statistics first, mechanistic model as backup:
  1. Per-day regression (data-driven), on log values, one model per variable and sampling day,
     using only what is known for a test batch: day-0 values, volume, Feed B / substrate-feed profile.
       - VCD and TCD: each day's model also learns from the neighbouring days (j-3 .. j+3)
       - substrate: ridge regression per day
       - byproduct: robust (Huber) regression per day, so late byproduct spikes don't pull the fit
  2. Blend with the typical curve of past batches (scaled to each batch's starting cells):  20 %
  3. Blend with the mechanistic ODE model (growth, death, inhibitor, overflow, feeds as boluses):
       VCD 20 %, TCD 20 %, substrate 10 %, byproduct 0 %
  v3 additions:
     - Feed C: besides 8 fixed recipe doses, batches get 3-27 small control-triggered doses; their count
       (known for test batches) tells how a batch behaves -> extra input for VCD, TCD and byproduct
     - Batches whose substrate top-up (Feed A) starts unusually early (before day 11): after that top-up
       the mechanistic model gets 40 % weight for substrate (it simulates the feed timing physically,
       the statistics have seen few such batches)
  Blends are geometric (on log scale); all weights were chosen by 5-fold cross-validation.
  Local cross-validation (mean MAPE): iteration 1 7.63 -> v2 7.40 -> v3 7.31.
  Per-day regression approach based on our teammate's model (iteration 1).

Run:    python main_v3.py
Output: submission_v3.csv (upload this), next to where you run it
"""
import sys, json, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

DATA_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv"
TEMPLATE_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_template.csv"
OUTPUT = "submission_v3.csv"

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


# =====================================================================================================
# Part 1: per-day regression (statistics)
# =====================================================================================================
T = ["vcd", "tcd", "substrate", "byproduct"]
ALPHAS = np.logspace(-2, 4, 25)
REDUCED = ["log_vcd0", "log_tcd0", "log_substrate0", "log_byproduct0", "V0", "feed_substrate_tot", "feed_b_tot",
           "fs_t_first", "fs_first_amt", "t", "Vt", "feed_substrate_cum_t", "feed_substrate_n_t",
           "feed_b_cum_t", "feed_b_n_t"]
FEAT = {"vcd": REDUCED + ["nCs_tot"], "tcd": REDUCED + ["nCs_tot"],
        "substrate": [c for c in REDUCED if c not in ("log_byproduct0", "feed_substrate_n_t", "log_vcd0")],
        "byproduct": [c for c in REDUCED if c not in ("log_substrate0", "fs_t_first", "fs_first_amt", "t",
                                                      "feed_substrate_n_t", "feed_substrate_cum_t")] + ["nCs_tot"]}
SMALL_C = 0.0013          # Feed C doses below this volume are the small, control-triggered ones
EARLY_TOPUP_DAY = 11.0    # first Feed A dose before this day = early top-up batch
W_MECH_SUBSTRATE_AFTER_EARLY_TOPUP = 0.4
WINDOW = {"vcd": 3, "tcd": 3, "substrate": 0, "byproduct": 0}    # neighbouring days each model learns from
W_BASELINE = 0.2                                                  # weight of the typical curve
W_MECH = {"vcd": 0.2, "tcd": 0.2, "substrate": 0.1, "byproduct": 0.0}   # weight of the mechanistic model


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def huber():
    return make_pipeline(StandardScaler(), HuberRegressor(alpha=0.3, epsilon=1.35, max_iter=1000))


MODEL = {"vcd": ridge, "tcd": ridge, "substrate": ridge, "byproduct": huber}


def build_table(d):
    """One row per batch and sampling time (> 0): features known for test batches + targets."""
    rows = []
    for b, g in d.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        m = g[g.vcd.notna() | g.to_predict].reset_index(drop=True)
        m0 = m.iloc[0]
        x = {f"log_{v}0": np.log(m0[v]) for v in T}
        x["V0"] = m0.volume
        for f in FEEDS:
            x[f + "_tot"] = g[f].sum()
        c = g[g.feed_c.notna()]
        x["nCs_tot"] = int((c.feed_c <= SMALL_C).sum())
        s = g[g.feed_substrate.notna()]
        x["fs_t_first"] = s.elapsed_time.iloc[0]
        x["fs_first_amt"] = s.feed_substrate.iloc[0]
        for j in range(1, len(m)):
            t = m.loc[j, "elapsed_time"]; h = g[g.elapsed_time <= t]
            r = dict(batch_id=b, j=j, elapsed_time=t, t=t, Vt=m.loc[j, "volume"], **x,
                     **{v: m.loc[j, v] for v in T})
            for f in FEEDS:
                r[f + "_cum_t"] = h[f].sum(); r[f + "_n_t"] = h[f].count()
            rows.append(r)
    return pd.DataFrame(rows)


def regression_predict(tr, te):
    """Per (variable, day) regression on log values; returns log predictions."""
    out = te[["batch_id", "elapsed_time"]].copy()
    for v in T:
        w, cols = WINDOW[v], FEAT[v]
        out[v] = np.nan
        for j, g in te.groupby("j"):
            gt = tr[(tr.j - j).abs() <= w]
            mu = gt.groupby("j")[v].apply(lambda s: np.log(s).mean())       # typical log value per day
            X = gt[cols].assign(dj=gt.j - j) if w else gt[cols]
            Xt = g[cols].assign(dj=0) if w else g[cols]
            model = MODEL[v]().fit(X, np.log(gt[v]) - gt.j.map(mu))
            out.loc[g.index, v] = model.predict(Xt) + mu.loc[j]
    return out


def baseline_predict(tr, te):
    """Typical curve of past batches (mean log value per day); cells scaled to the batch's day-0 value."""
    out = te[["batch_id", "elapsed_time"]].copy()
    for v in T:
        val = tr.groupby("j")[v].apply(lambda s: np.log(s).mean()).loc[te.j].values
        if v in ("vcd", "tcd"):
            val = val + (te[f"log_{v}0"].values - tr[f"log_{v}0"].mean())
        out[v] = val
    return out


# =====================================================================================================
# Run everything
# =====================================================================================================
if __name__ == "__main__":
    if len(sys.argv) > 2:
        DATA_PATH, TEMPLATE_PATH = sys.argv[1], sys.argv[2]
    d = pd.read_csv(DATA_PATH); tmpl = pd.read_csv(TEMPLATE_PATH)
    n = d.groupby("batch_id").vcd.count(); train = sorted(n[n > 1].index)
    test = list(dict.fromkeys(tmpl.batch_id))
    keys = tmpl.batch_id + "|" + tmpl.elapsed_time.round(6).astype(str)
    print(f"{len(train)} training batches, {len(test)} test batches", flush=True)

    print("1) per-day regression ...", flush=True)
    tab = build_table(d)
    tr, te = tab[tab.batch_id.isin(train)], tab[tab.batch_id.isin(test)]
    reg, base = regression_predict(tr, te), baseline_predict(tr, te)
    stats = te[["batch_id", "elapsed_time"]].copy()
    for v in T:
        stats[v] = (1 - W_BASELINE) * reg[v] + W_BASELINE * base[v]          # log scale
    stats["key"] = stats.batch_id + "|" + stats.elapsed_time.round(6).astype(str)
    stats = stats.set_index("key")
    assert keys.isin(stats.index).all(), "regression: some template rows missing"

    print("2) mechanistic model (fitting takes ~1-2 min) ...", flush=True)
    prep = prepare(d, train)
    p = {k: v[0] for k, v in PARAMS.items()}
    stages = [("cells", ["mu_max", "KA", "nA", "kdA", "KdA", "tlag", "kd0", "kd1", "Kd", "klys"], (0, 1), 40),
              ("substrate", ["m", "Yxs", "c_B", "c_A", "q_ov", "Kov", "n_ov", "Kx"], (2, 3), 25),
              ("byproduct", ["Yp", "q_c", "Kc"], (3,), 30),
              ("joint", NAMES, (0, 1, 2, 3), 30)]
    for name, free, use, nf in stages:
        t0 = time.time(); p = fit(prep, p, free, use, nf)
        print(f"   stage {name:9s} done in {time.time()-t0:4.0f}s", flush=True)
    json.dump(p, open("fitted_parameters.json", "w"), indent=1)
    prep_t = prepare(d, test)
    mech = pd.DataFrame(np.log(np.maximum(simulate(p, prep_t), 1e-3)), columns=OBS)
    mech["key"] = [f"{s[4]}|{round(s[1], 6)}" for s in prep_t["samples"]]
    mech = mech.set_index("key")
    assert keys.isin(mech.index).all(), "mechanistic: some template rows missing"

    print("3) combine and write submission ...", flush=True)
    first_topup = d[d.feed_substrate.notna()].groupby("batch_id").elapsed_time.min()
    f1 = first_topup.reindex(tmpl.batch_id).values
    after_early = (f1 < EARLY_TOPUP_DAY) & (tmpl.elapsed_time.values > f1)
    print(f"   early-top-up test batches: {sorted(tmpl.batch_id[after_early].unique())}", flush=True)
    out = tmpl.copy()
    for v in T:
        w = np.full(len(tmpl), W_MECH[v])
        if v == "substrate":
            w[after_early] = W_MECH_SUBSTRATE_AFTER_EARLY_TOPUP
        out[v] = np.exp((1 - w) * stats.loc[keys, v].values + w * mech.loc[keys, v].values)
    out["tcd"] = np.maximum(out.tcd, out.vcd)
    assert list(out.columns) == list(tmpl.columns) and len(out) == len(tmpl) and out[T].notna().all().all()
    assert (out[T] > 0).all().all()
    out.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT} ({len(out)} rows)  <- upload this one")
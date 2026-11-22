"""Team proposal: y(b,d) = median_d + lambda_t * (sim(b,d) - mean_d sim) + ridge(x_b) on the residuals (log scale).
sim = teammate ODE (src/v2.py) fitted fold by fold (the *_mech columns of their_oof, honest). Same folds as everything.
Evaluated standalone and as a blend member of the current best (LB 7.300 version). Choice on seeds 0-2.
Usage: python src/experiments/physics_deviation.py"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(SRC / "experiments"))
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import data as D, evaluation as E, regression as R, model_families as F
import controller_kalman as K, controller_kalman3 as K3

OUT = SRC.parent / "results" / "families"
raw = D.load_raw(); TRAIN = D.train_batches(raw)
m = D.measurements(raw); m["j"] = m.groupby("batch_id").cumcount(); truth = m.set_index(["batch_id", "j"])
T = ["vcd", "tcd", "substrate", "byproduct"]; J = list(range(1, 14))
tab = R.build_table(raw); st = tab[tab.j == 1].set_index("batch_id")
samp = D.measurements(raw, include_to_predict=True); samp["j"] = samp.groupby("batch_id").cumcount()
Vm = samp.pivot(index="batch_id", columns="j", values="volume")
X = pd.DataFrame({"log_vcd0": st.log_vcd0, "log_tcd0": st.log_tcd0, "log_substrate0": st.log_substrate0, "log_byproduct0": st.log_byproduct0,
                  "fs_tot": st.feed_substrate_tot, "fb_tot": st.feed_b_tot, "fc_tot": st.feed_c_tot, "fd_tot": st.feed_d_tot,
                  "vol_added": Vm[13] - Vm[0]})


def model(s, lam):
    th = pd.read_csv(OUT / f"their_oof_s{s}.csv").set_index(["batch_id", "j"])
    sim = {t: th[f"{t}_mech"].unstack()[J] for t in T}                                 # log simulations (OOF ODE)
    Y = {t: np.log(truth[t]).unstack().loc[sim[t].index, J] for t in T}
    out = {t: pd.DataFrame(index=sim[t].index, columns=J, dtype=float) for t in T}
    for fo in E.make_folds(TRAIN, 7, s):
        tr = [b for b in TRAIN if b not in fo]
        for t in T:
            mu = Y[t].loc[tr].median(); dev = sim[t] - sim[t].loc[tr].mean()
            for d in J:
                base = mu[d] + lam[t] * dev[d]
                r = Y[t].loc[tr, d] - base.loc[tr]
                rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(X.loc[tr], r)
                out[t].loc[fo, d] = base.loc[fo] + rg.predict(X.loc[fo])
    return {t: out[t].stack() for t in T}


def current_best(s):
    ou = pd.read_csv(OUT / f"ours_oof_s{s}.csv").set_index(["batch_id", "j"])[T]; idx = ou.index
    th = pd.read_csv(OUT / f"their_oof_s{s}.csv").set_index(["batch_id", "j"])[T].loc[idx]
    fam = {n: g.set_index(["batch_id", "j"])[T].loc[idx] for n, g in pd.read_csv(OUT / f"families_oof_s{s}.csv").groupby("model")}
    hb = pd.read_csv(OUT / f"byphuber_oof_s{s}.csv").set_index(["batch_id", "j"]).byproduct.loc[idx]
    L = {}
    for v, w in (("vcd", 0.6), ("tcd", 0.5)):
        team = (1 - w) * np.log(ou[v]) + w * np.log(th[v])
        L[v] = 0.25 * team + 0.55 * np.mean([np.log(fam[n][v]) for n in F.KERNEL_GROUP], axis=0) + 0.2 * np.mean([np.log(fam[n][v]) for n in F.TREE_GROUP], axis=0)
    L["tcd"] = np.maximum(L["tcd"], L["vcd"])
    L["substrate"] = np.log(K3.run(s, tau_late=0.03, tau_mid=0.1)).loc[idx]           # kalman3 full (LB 7.300)
    L["byproduct"] = 0.9 * np.log(hb) + 0.1 * np.log(th.byproduct)
    return L


def mape(lp, t):
    y = truth.loc[lp.index, t]
    return 100 * np.mean(np.abs(np.exp(lp) - y) / y)


def main():
    seeds = range(6)
    best = {s: current_best(s) for s in seeds}
    print("current best (LB 7.300):", {t: f"{np.mean([mape(best[s][t], t) for s in (0, 1, 2)]):.3f}/{np.mean([mape(best[s][t], t) for s in (3, 4, 5)]):.3f}" for t in T}, flush=True)
    for lam_v in (0.0, 0.25, 0.5, 0.75, 1.0):
        lam = {t: lam_v for t in T}
        P = {s: model(s, lam) for s in seeds}
        alone = {t: (np.mean([mape(P[s][t], t) for s in (0, 1, 2)]), np.mean([mape(P[s][t], t) for s in (3, 4, 5)])) for t in T}
        line = []
        for t in T:
            gains = []
            for w in (0.1, 0.2, 0.3):
                g = [mape(best[s][t], t) - mape((1 - w) * best[s][t] + w * P[s][t].loc[best[s][t].index], t) for s in seeds]
                gains.append((w, np.mean(g[:3]), np.mean(g[3:])))
            wb = max(gains, key=lambda x: x[1])
            line.append(f"{t}: alone {alone[t][0]:.2f}/{alone[t][1]:.2f}, blend w={wb[0]} gain {wb[1]:+.3f}/{wb[2]:+.3f}")
        print(f"lambda={lam_v}: " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()

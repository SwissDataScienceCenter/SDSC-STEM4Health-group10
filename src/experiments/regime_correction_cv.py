"""Residual correction by CONTROL REGIME (cross-fitted), on top of the current best blend + substrate rules.

The feed profile tells which control regime a batch went through: mid-run extra substrate bolus (and whether a
sample is before / at / after it), big or small late substrate dose after samples 11 and 12. The out-of-fold
errors of the current best predictions show systematic biases per regime (e.g. VCD under-predicted by 4-7 %
at days 5-6 in mid-run-bolus batches: more cells -> faster substrate depletion -> bolus).
For each target and sampling index j:  log-error ~ centred regime indicators (ridge, no intercept), fitted on the
OOF errors of the training batches of each fold and applied to the validation batches (cross-fitting).
Penalty chosen on CV seeds 0-2, checked on fresh seeds 3-5.
Usage (repo root): python src/experiments/regime_correction_cv.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
import control_rules as C
import data as D
import evaluation as E
import model_families as F

OUT = SRC.parent / "results" / "families"
T = ["vcd", "tcd", "substrate", "byproduct"]
WT = {"vcd": 0.6, "tcd": 0.5, "substrate": 0.5, "byproduct": 1.0}
SEL, CONF = [0, 1, 2], [3, 4, 5]


def regime_indicators(raw: pd.DataFrame, index: pd.MultiIndex) -> pd.DataFrame:
    trig, big = C.trigger_samples(raw), C.late_dose_classes(raw)
    b = index.get_level_values(0); j = index.get_level_values(1).to_numpy()
    j0 = b.map(trig).to_numpy(dtype=float)
    z = pd.DataFrame(index=index)
    z["trig_pre"] = (~np.isnan(j0)) & (j < j0)
    z["trig_post"] = (~np.isnan(j0)) & (j > j0)
    z["big11"] = [bool(big.get((bb, 11), False)) for bb in b]
    z["big12"] = [bool(big.get((bb, 12), False)) for bb in b]
    return z.astype(float)


def current_best(s, raw):
    ou = pd.read_csv(OUT / f"ours_oof_s{s}.csv").set_index(["batch_id", "j"])[T]
    idx = ou.index
    th = pd.read_csv(OUT / f"their_oof_s{s}.csv").set_index(["batch_id", "j"])[T].loc[idx]
    fam = {n: g.set_index(["batch_id", "j"])[T].loc[idx] for n, g in pd.read_csv(OUT / f"families_oof_s{s}.csv").groupby("model")}
    L = pd.DataFrame({v: (1 - WT[v]) * np.log(ou[v]) + WT[v] * np.log(th[v]) for v in T})
    for v in F.CELL_TARGETS:
        L[v] = (F.W_FINAL["team"] * L[v] + F.W_FINAL["kernel"] * np.mean([np.log(fam[n][v]) for n in F.KERNEL_GROUP], axis=0)
                + F.W_FINAL["trees"] * np.mean([np.log(fam[n][v]) for n in F.TREE_GROUP], axis=0))
    return L


def apply_rules(L, raw, tr, ids):
    d = np.exp(L.loc[ids]).reset_index()
    d = C.SubstrateControlRule().fit(raw, tr).apply(d)
    d = C.LateDoseRule().fit(raw, tr).apply(d)
    return np.log(d.set_index(["batch_id", "j"])[T])


def run(s, raw, truth, lam, targets=T, cols=None):
    TR = D.train_batches(raw)
    L = current_best(s, raw)
    Z = regime_indicators(raw, L.index)
    if cols is not None:
        Z = Z[cols]
    Y = np.log(truth.loc[L.index, T])
    out = []
    for fo in E.make_folds(TR, 7, s):
        tr = [b for b in TR if b not in fo]
        tr_m = L.index.get_level_values(0).isin(tr)
        Lc = L.copy()
        for v in targets:
            if lam is None:
                continue
            for j in range(1, 14):
                a = tr_m & (L.index.get_level_values(1) == j)
                te = (~tr_m) & (L.index.get_level_values(1) == j)
                Za = Z[a].to_numpy(); mu = Za.mean(axis=0); Zc = Za - mu
                e = (L[v][a] - Y[v][a]).to_numpy()
                beta = np.linalg.solve(Zc.T @ Zc + lam * np.eye(Zc.shape[1]), Zc.T @ e)
                Lc.loc[te, v] = L[v][te] - (Z[te].to_numpy() - mu) @ beta
        out.append(apply_rules(Lc, raw, tr, fo))
    P = pd.concat(out)
    y = truth.loc[P.index, T]
    return {v: 100 * np.mean(np.abs(np.exp(P[v]) - y[v]) / y[v]) for v in T}


def main():
    raw = D.load_raw()
    m = D.measurements(raw); m["j"] = m.groupby("batch_id").cumcount()
    truth = m.set_index(["batch_id", "j"])
    base = pd.DataFrame([run(s, raw, truth, None) for s in SEL + CONF], index=SEL + CONF)
    print("current best + rules (no correction):", base.mean().round(3).to_dict())
    for lam in [1, 3, 10, 30]:
        r = pd.DataFrame([run(s, raw, truth, lam) for s in SEL + CONF], index=SEL + CONF)
        g = base - r
        print(f"lambda={lam:3d}: gain seeds 0-2 {g.loc[SEL].mean().round(3).to_dict()} | fresh 3-5 {g.loc[CONF].mean().round(3).to_dict()}"
              f" | mean over targets 0-2 {g.loc[SEL].mean().mean():+.3f}, 3-5 {g.loc[CONF].mean().mean():+.3f}", flush=True)


if __name__ == "__main__":
    main()

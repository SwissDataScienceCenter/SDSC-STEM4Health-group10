"""Wave-2 analysis: per target, standalone scores and greedy ensemble selection (Caruana et al. 2004) over all
out-of-fold members (team models, wave-1 families, wave-2 variants), starting from the current best submission
(blend_nonlinear). Members are chosen on CV seeds 0-2 and the result is checked on fresh seeds 3-5.
Usage (repo root): python src/experiments/wave2_analysis.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
import data as D
import model_families as F

OUT = SRC.parent / "results" / "families"
T = ["vcd", "tcd", "substrate", "byproduct"]
W_TEAM = {"vcd": 0.6, "tcd": 0.5, "substrate": 0.5, "byproduct": 1.0}
SEL, CONF = [0, 1, 2], [3, 4, 5]
SEEDS = SEL + CONF
N_PRIOR, N_STEPS = 4, 16          # ensemble starts with 4 copies of the current best, then 16 greedy additions

m = D.measurements(D.load_raw())
m["j"] = m.groupby("batch_id").cumcount()
truth = m.set_index(["batch_id", "j"])[T]
L = {}                                                     # (member, seed) -> log predictions (DataFrame)
for s in SEEDS:
    ou = pd.read_csv(OUT / f"ours_oof_s{s}.csv").set_index(["batch_id", "j"])[T]
    idx = ou.index
    th = pd.read_csv(OUT / f"their_oof_s{s}.csv").set_index(["batch_id", "j"])[T].loc[idx]
    L["ours", s], L["theirs", s] = np.log(ou), np.log(th)
    for f in ("families", "wave2"):
        p = OUT / f"{f}_oof_s{s}.csv"
        if p.exists():
            for name, g in pd.read_csv(p).groupby("model"):
                if not name.startswith("mlp"):
                    L[name, s] = np.log(g.set_index(["batch_id", "j"])[T].loc[idx])
    team = pd.DataFrame({v: (1 - W_TEAM[v]) * L["ours", s][v] + W_TEAM[v] * L["theirs", s][v] for v in T})
    team["tcd"] = np.maximum(team.tcd, team.vcd)
    L["blend_team", s] = team
    best = team.copy()
    for v in F.CELL_TARGETS:
        kern = np.mean([L[n, s][v] for n in F.KERNEL_GROUP], axis=0)
        trees = np.mean([L[n, s][v] for n in F.TREE_GROUP], axis=0)
        best[v] = F.W_FINAL["team"] * team[v] + F.W_FINAL["kernel"] * kern + F.W_FINAL["trees"] * trees
    L["current_best", s] = best

members = sorted({k for k, _ in L})
tv = {s: truth.loc[L["ours", s].index] for s in SEEDS}


def mape(logp, v, s):
    y = tv[s][v].to_numpy()
    return 100 * np.mean(np.abs(np.exp(logp) - y) / y)


alone = pd.DataFrame({mo: {v: np.mean([mape(L[mo, s][v].to_numpy(), v, s) for s in SEEDS]) for v in T} for mo in members}).T
alone["mean"] = alone[T].mean(axis=1)
print("STANDALONE MAPE (%), seeds 0-5 (best 12)\n", alone.sort_values("mean").head(12).round(3).to_string())
for v in ("substrate", "byproduct"):
    print(f"best for {v}:", alone[v].sort_values().head(6).round(3).to_dict())

res = {}
for v in T:
    ens = ["current_best"] * N_PRIOR
    arr = {(mo, s): L[mo, s][v].to_numpy() for mo in members for s in SEEDS}
    score = lambda e, ss: np.mean([mape(np.mean([arr[mo, s] for mo in e], axis=0), v, s) for s in ss])
    for _ in range(N_STEPS):
        cand = min(members, key=lambda mo: score(ens + [mo], SEL))
        if score(ens + [cand], SEL) >= score(ens, SEL) - 1e-4:
            break
        ens.append(cand)
    w = pd.Series(ens).value_counts() / len(ens)
    res[v] = dict(base_sel=score(["current_best"], SEL), ens_sel=score(ens, SEL),
                  base_conf=score(["current_best"], CONF), ens_conf=score(ens, CONF), weights=w.round(3).to_dict())
    print(f"\n--- {v}: current best {res[v]['base_conf']:.3f} -> ensemble {res[v]['ens_conf']:.3f} on fresh seeds 3-5 "
          f"(seeds 0-2: {res[v]['base_sel']:.3f} -> {res[v]['ens_sel']:.3f})\n    weights: {res[v]['weights']}")
gain = np.mean([res[v]["base_conf"] - res[v]["ens_conf"] for v in T])
print(f"\nmean gain over the 4 targets on fresh seeds 3-5: {gain:.3f} MAPE points")
pd.DataFrame(res).T.to_csv(OUT / "wave2_ensemble.csv")
alone.to_csv(OUT / "wave2_standalone.csv")

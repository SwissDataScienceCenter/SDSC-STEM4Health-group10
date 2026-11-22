"""Per-target comparison of the model families vs the team blend (blend_team), from the OOF files written by
src/experiments/families_cv.py. Blend weights are chosen on CV seeds 0-2 and confirmed on fresh seeds 3-5.
Usage (repo root): python src/experiments/families_analysis.py
Outputs: results/families/families_standalone.csv, families_blend_gains.csv, final_blend_gain.csv
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
W_TEAM = {"vcd": 0.6, "tcd": 0.5, "substrate": 0.5, "byproduct": 1.0}      # src/blend_team.py
SEL, CONF = [0, 1, 2], [3, 4, 5]
SEEDS = SEL + CONF

m = D.measurements(D.load_raw())
m["j"] = m.groupby("batch_id").cumcount()
truth = m.set_index(["batch_id", "j"])[T]
pred = {}                                                  # (model, seed) -> predictions indexed (batch_id, j)
for s in SEEDS:
    ou = pd.read_csv(OUT / f"ours_oof_s{s}.csv").set_index(["batch_id", "j"])[T]
    th = pd.read_csv(OUT / f"their_oof_s{s}.csv").set_index(["batch_id", "j"])[T].loc[ou.index]
    ref = pd.DataFrame({v: np.exp((1 - W_TEAM[v]) * np.log(ou[v]) + W_TEAM[v] * np.log(th[v])) for v in T})
    ref["tcd"] = np.maximum(ref.tcd, ref.vcd)
    pred["ours", s], pred["theirs", s], pred["blend_team", s] = ou, th, ref
    for name, g in pd.read_csv(OUT / f"families_oof_s{s}.csv").groupby("model"):
        pred[name, s] = g.set_index(["batch_id", "j"])[T].loc[ou.index]


def mape(p: pd.Series, v: str) -> float:
    return 100 * np.mean(np.abs(p - truth.loc[p.index, v]) / truth.loc[p.index, v])


def geo(parts):                                            # weighted geometric mean of (weight, series)
    return np.exp(sum(w * np.log(x) for w, x in parts))


models = sorted({k for k, _ in pred})
alone = pd.DataFrame({mo: {v: np.mean([mape(pred[mo, s][v], v) for s in SEEDS]) for v in T} for mo in models}).T
alone["mean"] = alone[T].mean(axis=1)
alone.sort_values("mean").to_csv(OUT / "families_standalone.csv")
print("STANDALONE MAPE (%), mean over seeds 0-5\n", alone.sort_values("mean").round(3).to_string(), "\n")

# 2-member blends: blend_team + one model, weight chosen on seeds 0-2
rows = []
for mo in models:
    if mo == "blend_team":
        continue
    for v in T:
        sc = lambda w, ss: np.mean([mape(geo([(1 - w, pred["blend_team", s][v]), (w, pred[mo, s][v])]), v) for s in ss])
        w = min([0.1, 0.2, 0.3, 0.4, 0.5], key=lambda w: sc(w, SEL))
        rows.append({"model": mo, "target": v, "w": w, "gain_seeds0-2": sc(0, SEL) - sc(w, SEL),
                     "gain_seeds3-5": sc(0, CONF) - sc(w, CONF)})
r = pd.DataFrame(rows)
r.to_csv(OUT / "families_blend_gains.csv", index=False)
for v in T:
    print(f"--- {v}: best additions to blend_team (gain in MAPE points, + = better)")
    print(r[r.target == v].sort_values("gain_seeds3-5", ascending=False).head(5).round(3).to_string(index=False))

# Final 3-member blend for vcd / tcd (src/model_families.py: W_FINAL)
grp = lambda names, s, v: np.exp(np.mean([np.log(pred[n, s][v]) for n in names], axis=0))
W = F.W_FINAL
fin = []
for v in F.CELL_TARGETS:
    for s in SEEDS:
        p = pd.Series(geo([(W["team"], pred["blend_team", s][v]), (W["kernel"], grp(F.KERNEL_GROUP, s, v)),
                           (W["trees"], grp(F.TREE_GROUP, s, v))]), index=pred["blend_team", s].index)
        fin.append({"target": v, "seed": s, "blend_team": mape(pred["blend_team", s][v], v), "blend_nonlinear": mape(p, v)})
fin = pd.DataFrame(fin)
fin["gain"] = fin.blend_team - fin.blend_nonlinear
fin.to_csv(OUT / "final_blend_gain.csv", index=False)
print("\nFINAL blend (team 0.25 + kernel 0.55 + trees 0.20) vs blend_team, per seed:\n", fin.round(3).to_string(index=False))
print("\nmean gain on fresh seeds 3-5:", fin[fin.seed.isin(CONF)].groupby("target").gain.mean().round(3).to_dict())

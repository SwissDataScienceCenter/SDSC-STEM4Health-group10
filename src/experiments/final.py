"""Final combined model of the overnight session: per-target geometric blends of LOBO-validated members.

CONFIG[name][target] = {member: weight}. Weights = the nested-CV ensemble's weights refitted on all 49 batches
(src/experiments/ensemble.py) for the targets where the ensemble won; the reference for the others.
The honest LOBO estimate of each config is assembled from the NESTED out-of-fold predictions (see HONEST_OOF).

Usage: python src/experiments/final.py NAME     -> results/overnight/NAME_score.csv + submissions/NAME.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as X  # noqa: E402
import members as M  # noqa: E402

FACTORIES = {**M.MEMBERS, "ref_ridge_blend": X.ridge_blend}

REF = {"ref_ridge_blend": 1.0}
CONFIG = {
    # v1: cell densities = nested ensemble of ridge (1 FPCA mode, penalty x0.25) + nearest batches (+ baseline);
    #     substrate / byproduct = reference ridge_blend unchanged.
    "overnight_v1": {
        "vcd": {"ridge_s_k025": 0.5, "nearest": 0.5},
        "tcd": {"ridge_s_k025": 0.5, "base_scaled": 0.1, "nearest": 0.4},
        "substrate": REF,
        "byproduct": REF,
    },
    # v2 (simpler, preferred): plain 50/50 geometric mean for both cell densities = what every nested run converged to
    # (ens_k025+nearestK, ens_k025+nearestK2, ens_full all refit to exactly 0.5 / 0.5 on vcd and tcd).
    "overnight_v2": {
        "vcd": {"ridge_s_k025": 0.5, "nearest": 0.5},
        "tcd": {"ridge_s_k025": 0.5, "nearest": 0.5},
        "substrate": REF,
        "byproduct": REF,
    },
}
# per target: which saved OOF gives the honest (nested) LOBO estimate of that target's predictions
HONEST_OOF = {
    "overnight_v1": {"vcd": "ens_ridge_s+ridge_s_k025+base_scaled+nearest",
                     "tcd": "ens_ridge_s+ridge_s_k025+base_scaled+nearest",
                     "substrate": "ref_ridge_blend", "byproduct": "ref_ridge_blend"},
    # conservative: the 4-member nested OOF (the 3-member nested runs give 6.564 / 6.493, slightly better)
    "overnight_v2": {"vcd": "ens_ridge_s+ridge_s_k025+base_scaled+nearest",
                     "tcd": "ens_ridge_s+ridge_s_k025+base_scaled+nearest",
                     "substrate": "ref_ridge_blend", "byproduct": "ref_ridge_blend"},
}


def predict(cfg: dict, train_ids, ids) -> pd.DataFrame:
    needed = {m for w in cfg.values() for m in w}
    preds = {m: X._sorted(FACTORIES[m](train_ids, ids)) for m in needed}
    out = next(iter(preds.values()))[["batch_id", "j", "elapsed_time"]].copy()
    for v, w in cfg.items():
        assert abs(sum(w.values()) - 1) < 1e-9
        out[v] = np.exp(sum(wi * np.log(preds[m][v].to_numpy()) for m, wi in w.items()))
    return out


def honest_oof(name: str) -> pd.DataFrame:
    src = HONEST_OOF[name]
    oofs = {n: X._sorted(pd.read_csv(X.OUT / f"{n}_oof.csv")) for n in set(src.values())}
    out = X.truth()[["batch_id", "j", "elapsed_time"]].copy()
    for v, n in src.items():
        out[v] = oofs[n][v].to_numpy()
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    X.report(name, honest_oof(name))
    _, _, train, test = X.load()
    X.write_submission(name, predict(CONFIG[name], train, test))

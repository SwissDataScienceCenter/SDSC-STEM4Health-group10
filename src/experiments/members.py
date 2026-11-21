"""Unblended ensemble members (LOBO OOF), so that the baseline weight is tuned by nested CV instead of fixed at 0.2.

ridge_s      : per-(target, j) RidgeCV on log targets, reference features, vcd/tcd projected on mean + 1 FPCA mode
               (the nested choice of smooth_pls.py), substrate/byproduct not projected. NOT blended.
ridge_s_k025 : same with the penalty x 0.25 (weaker shrinkage; R.ScaledRidgeCV).
base_scaled  : VCD0-scaled mean training curve (src/baselines.py).
nearest      : geometric mean of the 8 nearest training batches (R.NearestBatches).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as X  # noqa: E402

R = X.R
SMOOTH = {"vcd": 1, "tcd": 1, "substrate": None, "byproduct": None}


def ridge_s(k=1.0):
    def fp(train_ids, ids):
        table = X.load()[1]
        make = R.ridge if k == 1.0 else R.scaled_ridge(k)
        m = R.PerTimeRegression(R.FEATURES_BY_TARGET, make, n_eigencurves=SMOOTH)
        return X._sorted(m.fit(table, train_ids).predict(table, ids))
    return fp


def nearest(train_ids, ids):
    table = X.load()[1]
    return X._sorted(R.NearestBatches().fit(table, train_ids).predict(table, ids))


MEMBERS = {"ridge_s": ridge_s(1.0), "ridge_s_k025": ridge_s(0.25), "base_scaled": X.baseline_scaled, "nearest": nearest}

if __name__ == "__main__":
    for name, fp in MEMBERS.items():
        X.report(name, X.lobo(fp))

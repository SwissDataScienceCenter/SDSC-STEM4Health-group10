"""Team's hybrid ODE (src/ode_paramnet.py: model v3, 6 batch-specific parameters predicted by ridge from batch
features) as a LOBO member for the cell-density ensemble. Folds run in parallel processes.
Caveat: the SHARED v3 parameters (results/v3/params_full.csv) were fitted once on all 49 training batches; only the
per-batch fit + param-net are refitted per fold (team measured the effect as negligible: 7.014 vs 7.02 % VCD).
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as X  # noqa: E402
import ode_paramnet as O  # noqa: E402

SHARED = pd.read_csv(X.REPO / "results" / "v3" / "params_full.csv", index_col=0)["estimate"].to_dict()


def fit_predict(train_ids, ids):
    raw, table, _, _ = X.load()
    o = O.predict_ode(raw, table, train_ids, ids, SHARED, X.R.ANALOG_FEATURES)
    o = o.sort_values(["batch_id", "elapsed_time"]).reset_index(drop=True)
    keys = X._sorted(table[table.batch_id.isin(ids)][["batch_id", "j", "elapsed_time"]])
    o = keys.assign(_t=keys.elapsed_time.round(6)).merge(
        o.assign(_t=o.elapsed_time.round(6))[["batch_id", "_t", *X.TARGETS]], on=["batch_id", "_t"], how="left")
    o = o.drop(columns="_t")
    for v in X.TARGETS:
        o[v] = np.maximum(o[v].to_numpy(float), 1e-3)
    return o


def _fold(h):
    _, _, train, _ = X.load()
    return fit_predict([b for b in train if b != h], [h])


if __name__ == "__main__":
    _, _, train, _ = X.load()
    if sys.argv[1:] == ["time"]:
        t0 = time.time(); _fold(train[0]); print(f"one fold: {time.time() - t0:.1f} s"); sys.exit()
    with Pool(int(sys.argv[1]) if sys.argv[1:] else 4) as p:
        oof = pd.concat(p.map(_fold, train), ignore_index=True)
    X.report("ode_v3", X._sorted(oof))

"""Fit a mechanistic model on all training batches + batch-wise cross-validation, in parallel.

Usage (from the repo root):
    python src/run_cv.py v2_logistic [--folds 7] [--seed 0]

Outputs in results/<model>/:
    params_full.csv      parameters fitted on all 49 training batches (with indicative 95% CI)
    oof_predictions.csv  out-of-fold predictions (each batch predicted by a model that never saw it)
    cv_metrics.csv       RMSE / relative RMSE / MARE per variable
    fold_params.csv      parameters of each fold (stability check)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import evaluation as E
import mechanistic as M

REPO = Path(__file__).resolve().parents[1]


def _job(args):
    key, train, val, init, loss, residual, f_scale, tag = args
    raw = D.load_raw()
    t = time.time()
    r = M.fit(M.MODELS[key], raw, train, init=init, loss=loss, residual=residual, f_scale=f_scale,
              progress_every=500, tag=f"[{key} {tag}]")
    pred = M.predict(M.MODELS[key], raw, val, r.params) if val else None
    el = time.time() - t
    print(f"[{key} {tag}] DONE: fit on {len(train)} batches in {el:.0f}s (cost {r.cost:.1f})", flush=True)
    r.model = None          # keep the result picklable (sent back to the main process)
    return r, pred, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(M.MODELS))
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="JSON file with initial parameter values")
    ap.add_argument("--loss", default="linear", choices=["linear", "soft_l1"], help="least squares or robust loss")
    ap.add_argument("--residual", default="scaled", choices=["scaled", "log"],
                    help="scaled: (pred-meas)/std ; log: log(pred)-log(meas), relative errors (MAPE-aligned)")
    ap.add_argument("--f-scale", type=float, default=1.0, help="soft_l1 transition scale (0.1 = 10%% with log residuals)")
    ap.add_argument("--name", default=None, help="name of the output folder (default: model key [+ loss])")
    a = ap.parse_args()

    raw = D.load_raw()
    train_ids = D.train_batches(raw)
    folds = E.make_folds(train_ids, a.folds, a.seed)
    init = json.load(open(a.init)) if a.init else None
    default_name = a.model + ("" if a.residual == "scaled" else f"_{a.residual}") + ("" if a.loss == "linear" else f"_{a.loss}")
    out = REPO / "results" / (a.name or default_name)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(a.model, train_ids, [], init, a.loss, a.residual, a.f_scale, "full")]
    jobs += [(a.model, [b for b in train_ids if b not in f], f, init, a.loss, a.residual, a.f_scale, f"fold{i}")
             for i, f in enumerate(folds)]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        res = list(ex.map(_job, jobs))

    full = res[0][0]
    full.table.to_csv(out / "params_full.csv")
    print(f"[{out.name}] full fit: cost={full.cost:.1f} nfev={full.nfev} success={full.success} ({res[0][2]:.0f}s)")
    print(full.table[["estimate", "ci95_low", "ci95_high", "at_bound"]].round(4).to_string())

    oof = pd.concat([p.assign(fold=i) for i, (_, p, _) in enumerate(res[1:])], ignore_index=True)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame({i: r.params for i, (r, _, _) in enumerate(res[1:])}).T.to_csv(out / "fold_params.csv")
    truth = E.attach_truth(oof, D.measurements(raw))
    met = E.metrics(truth)
    met.to_csv(out / "cv_metrics.csv")
    print(f"\n[{out.name}] cross-validation ({a.folds} folds by batch):")
    print(met.round(3).to_string())
    mape = E.mape_summary(truth)
    mape.to_csv(out / "cv_mape.csv")
    print(f"\n[{out.name}] MAPE (leaderboard metric):")
    print(mape.round(2).to_string())


if __name__ == "__main__":
    main()

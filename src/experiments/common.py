"""Shared harness for the overnight experiments: leave-one-batch-out (LOBO) CV, scoring and submission.

Every experiment uses the SAME splits: each of the 49 training batches is held out once, the model is fitted on
the other 48 and predicts the held-out batch from its t = 0 state + feed profile + volume only (the feature table
of src/regression.py is built from exactly that information).

Usage in an experiment script:
    import common as X
    raw, table, train_ids, test_ids = X.load()
    oof = X.lobo(lambda tr, te: my_fit_predict(tr, te))   # DataFrame batch_id, j, elapsed_time, vcd, tcd, substrate, byproduct
    X.report("my_name", oof)                              # prints + saves results/overnight/<name>_{oof,score}.csv
    X.write_submission("my_name", my_fit_predict(train_ids, test_ids))

`fit_predict(train_ids, ids)` must return one row per (batch, sampling index j >= 1) of `ids`, with strictly
positive predictions of the 4 targets (original scale, not log).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
sys.path.insert(0, str(SRC))
import baselines as B  # noqa: E402
import data as D  # noqa: E402
import regression as R  # noqa: E402

warnings.filterwarnings("ignore")
TARGETS = R.TARGETS
OUT = REPO / "results" / "overnight"
OUT.mkdir(parents=True, exist_ok=True)

_CACHE: dict = {}


def load():
    """raw event log, feature table (train + test batches), train ids, test ids (cached)."""
    if not _CACHE:
        raw = D.load_raw()
        _CACHE.update(raw=raw, table=R.build_table(raw), train=D.train_batches(raw), test=D.test_batches(raw))
    return _CACHE["raw"], _CACHE["table"], _CACHE["train"], _CACHE["test"]


def _sorted(p: pd.DataFrame) -> pd.DataFrame:
    return p.sort_values(["batch_id", "elapsed_time"]).reset_index(drop=True)


def lobo(fit_predict) -> pd.DataFrame:
    """Leave-one-batch-out out-of-fold predictions over the 49 training batches."""
    _, _, train, _ = load()
    return _sorted(pd.concat([fit_predict([b for b in train if b != h], [h]) for h in train], ignore_index=True))


def truth() -> pd.DataFrame:
    _, table, train, _ = load()
    return _sorted(table[table.batch_id.isin(train)][["batch_id", "j", "elapsed_time", *TARGETS]])


def score(oof: pd.DataFrame) -> pd.Series:
    """MAPE (%) per target (all points pooled, as the leaderboard) + plain mean over the 4 targets."""
    t = truth()
    o = _sorted(oof)
    assert (o[["batch_id", "j"]].to_numpy() == t[["batch_id", "j"]].to_numpy()).all(), "rows misaligned"
    s = pd.Series({v: 100 * np.mean(np.abs(o[v] - t[v]) / t[v]) for v in TARGETS})
    s["mean"] = s[TARGETS].mean()
    return s


def report(name: str, oof: pd.DataFrame) -> pd.Series:
    s = score(oof)
    _sorted(oof)[["batch_id", "j", "elapsed_time", *TARGETS]].to_csv(OUT / f"{name}_oof.csv", index=False)
    s.rename(name).to_csv(OUT / f"{name}_score.csv")
    print(f"[{name}] LOBO MAPE (%): " + "  ".join(f"{k}={v:.3f}" for k, v in s.items()), flush=True)
    return s


def geo_blend(a: pd.DataFrame, b: pd.DataFrame, w) -> pd.DataFrame:
    """Geometric blend: exp((1-w) log a + w log b); w a float or a dict target -> weight of b."""
    a, b = _sorted(a), _sorted(b)
    assert (a[["batch_id", "j"]].to_numpy() == b[["batch_id", "j"]].to_numpy()).all()
    out = a.copy()
    for v in TARGETS:
        wv = w[v] if isinstance(w, dict) else w
        out[v] = np.exp((1 - wv) * np.log(a[v]) + wv * np.log(b[v]))
    return out


def baseline_scaled(train_ids, ids) -> pd.DataFrame:
    """VCD0-scaled mean training curve (src/baselines.py), with the j index attached."""
    raw, table, _, _ = load()
    b = _sorted(B.MeanTrajectory(scale_cells=True).fit(raw, train_ids).predict(raw, ids))
    keys = _sorted(table[table.batch_id.isin(ids)][["batch_id", "j", "elapsed_time"]])
    assert np.allclose(b.elapsed_time, keys.elapsed_time)
    return keys.assign(**{v: b[v].to_numpy() for v in TARGETS})


def ridge_blend(train_ids, ids, features=None, blend=0.2, fpca=False, tab=None) -> pd.DataFrame:
    """Reference model = submissions/ridge_blend.csv: per-(target, j) RidgeCV on log targets with per-target
    features (R.FEATURES_BY_TARGET), geometric 0.2 blend with the VCD0-scaled baseline.
    fpca=True adds the later FPCA projection of vcd/tcd (2 modes); `tab` = optional extended feature table."""
    table = load()[1] if tab is None else tab
    m = R.PerTimeRegression(features or R.FEATURES_BY_TARGET, R.ridge, n_eigencurves=R.N_EIGENCURVES if fpca else None)
    p = _sorted(m.fit(table, train_ids).predict(table, ids))
    return geo_blend(p, baseline_scaled(train_ids, ids), blend)


def write_submission(name: str, pred: pd.DataFrame) -> Path:
    """Template row order; validated: 156 rows, no NaN, strictly positive."""
    tpl = D.load_template()
    k = lambda d: d.assign(_t=d["elapsed_time"].round(6))
    sub = (k(tpl)[["batch_id", "elapsed_time", "_t"]]
           .merge(k(pred)[["batch_id", "_t", *TARGETS]], on=["batch_id", "_t"], how="left").drop(columns="_t"))
    assert len(sub) == 156 and sub[TARGETS].notna().all().all() and (sub[TARGETS] > 0).all().all()
    assert (sub[["batch_id", "elapsed_time"]].to_numpy() == tpl[["batch_id", "elapsed_time"]].to_numpy()).all()
    path = REPO / "submissions" / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"[{name}] submission written: {path.relative_to(REPO)}")
    return path


if __name__ == "__main__":
    # Reference score of the current best model under LOBO
    raw, table, train, test = load()
    report("ref_ridge_blend", lobo(ridge_blend))
    report("ref_ridge_blend_fpca", lobo(lambda tr, te: ridge_blend(tr, te, fpca=True)))
    # sanity: the reference reproduces submissions/ridge_blend.csv
    sub = ridge_blend(train, test)
    old = pd.read_csv(REPO / "submissions" / "ridge_blend.csv")
    k = lambda d: d.assign(_t=d["elapsed_time"].round(6)).set_index(["batch_id", "_t"])[TARGETS]
    print("max rel. diff vs submissions/ridge_blend.csv:",
          (k(sub).sort_index() / k(old).sort_index() - 1).abs().max().round(6).to_dict())

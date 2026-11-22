"""Experiment 4 (smoothing across time) + cheap PLS member for the ensemble.

(a) FPCA smoothing: the 13 per-time ridge predictions of a batch are projected onto the mean training log
    trajectory + its K main modes of variation (K in 1..4, same K for the 4 targets in one run), then the usual
    0.2 geometric blend with the scaled baseline. The K of each target is then chosen by NESTED selection
    (see nested_select).
(b) PLS: one PLSRegression per (target, j) on log targets, same per-target features, blended 0.2.

Nested selection on OOF predictions: for each held-out batch b, the candidate with the lowest MAPE on the OTHER
48 batches' OOF predictions is chosen, and its prediction for b is used. (The other batches' OOF predictions come
from models that saw b in training: selection-level only, much less optimistic than picking the best on LOBO.)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as X  # noqa: E402

R = X.R


def fpca_model(K):
    def fp(train_ids, ids):
        _, table, _, _ = X.load()
        m = R.PerTimeRegression(R.FEATURES_BY_TARGET, R.ridge, n_eigencurves={v: K for v in X.TARGETS})
        p = X._sorted(m.fit(table, train_ids).predict(table, ids))
        return X.geo_blend(p, X.baseline_scaled(train_ids, ids), 0.2)
    return fp


def pls_model(nc):
    def fp(train_ids, ids):
        _, table, _, _ = X.load()
        m = R.PerTimeRegression(R.FEATURES_BY_TARGET, lambda: make_pipeline(StandardScaler(), PLSRegression(nc, scale=False)))
        p = m.fit(table, train_ids)
        # PLSRegression.predict returns (n, 1): flatten through a thin wrapper
        p = X._sorted(_predict_flat(m, table, ids))
        return X.geo_blend(p, X.baseline_scaled(train_ids, ids), 0.2)
    return fp


def _predict_flat(m, table, ids):
    te = table[table.batch_id.isin(ids)]
    out = te[R.KEYS].copy()
    for v in X.TARGETS:
        out[v] = np.nan
        for j, g in te.groupby("j"):
            out.loc[g.index, v] = np.ravel(m.models[v, j].predict(g[m._cols(v)]))
        out[v] = np.exp(out[v])
    return out.reset_index(drop=True)


def batch_ape(oof: pd.DataFrame) -> pd.DataFrame:
    """Absolute percentage error per point, aligned with X.truth()."""
    t, o = X.truth(), X._sorted(oof)
    return pd.DataFrame({v: 100 * np.abs(o[v] - t[v]) / t[v] for v in X.TARGETS}).assign(batch_id=t.batch_id)


def nested_select(cands: dict) -> tuple[pd.DataFrame, dict]:
    """Per target and per held-out batch, pick the candidate that is best on the other 48 batches."""
    names = list(cands)
    ape = {n: batch_ape(cands[n]) for n in names}
    base = X._sorted(cands[names[0]]).copy()
    picks = {v: [] for v in X.TARGETS}
    for v in X.TARGETS:
        S = pd.DataFrame({n: ape[n].groupby("batch_id")[v].sum() for n in names})   # sum of APE per batch
        cnt = ape[names[0]].groupby("batch_id")[v].count()
        for b in S.index:
            n_best = ((S.drop(b).sum()) / cnt.drop(b).sum()).idxmin()
            picks[v].append(n_best)
            rows = base.batch_id == b
            base.loc[rows, v] = X._sorted(cands[n_best]).loc[rows, v].to_numpy()
    return base, {v: pd.Series(p).value_counts().to_dict() for v, p in picks.items()}


if __name__ == "__main__":
    cands = {"ref": pd.read_csv(X.OUT / "ref_ridge_blend_oof.csv")}
    for K in (1, 2, 3, 4):
        cands[f"fpca{K}"] = X.lobo(fpca_model(K))
        X.report(f"fpca_K{K}", cands[f"fpca{K}"])
    for nc in (2, 3, 5):
        X.report(f"pls{nc}", X.lobo(pls_model(nc)))
    sel, picks = nested_select(cands)
    X.report("fpca_nested", sel)
    print("nested picks per target:", picks)

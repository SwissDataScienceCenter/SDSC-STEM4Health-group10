"""Wave 2 of model experiments (all 4 targets), same folds as the team models (7 folds by batch, given seed).

Variants (target = log(y) - mu_j, the deviation from the mean log trajectory at sampling index j):
  svr_tw{w} / krr_tw{w} : pooled-time kernel models where the (standardised) time input is multiplied by w:
                          the kernel then mostly compares rows of NEARBY days -> "local-in-time pooling", a
                          smooth version of the teammate's neighbour-day pooling (w = 1: plain pooled model).
  vcr_k{K}              : varying-coefficient ridge: every feature x B-spline basis of time (K functions),
                          i.e. coefficients that change smoothly with time (between per-time and pooled).
  rs_svr                : residual stacking: per-time ridge (our model) + pooled SVR fitted on its INNER
                          out-of-fold residuals (corrects systematic errors, keeps the per-time structure).
  pt_svr                : per-time SVR (one SVR per sampling index, like the per-time ridge).
Feature sets: "sel" (per-target ridge selection + t), "ev" (sel + feed events since the previous sample:
substrate / Feed B added, time since the last substrate bolus).
Usage: python src/experiments/wave2_cv.py SEED  -> results/families/wave2_oof_s<SEED>.csv
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler
from sklearn.svm import SVR
import data as D
import evaluation as E
import regression as R

OUT = SRC.parent / "results" / "families"


def add_event_features(table: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Feed events between the previous sample and t (known from the feed profile of every batch)."""
    t = table.sort_values(["batch_id", "j"]).copy()
    for f in ("feed_substrate", "feed_b"):
        t[f"{f}_d_t"] = t.groupby("batch_id")[f"{f}_cum_t"].diff().fillna(t[f"{f}_cum_t"])
    fs = D.feeds_long(raw)
    fs = fs[fs.feed == "feed_substrate"]
    last = []
    for b, tt in zip(t.batch_id, t.elapsed_time):
        prev = fs.loc[(fs.batch_id == b) & (fs.elapsed_time <= tt), "elapsed_time"]
        last.append(min(tt - prev.max(), 3.0) if len(prev) else 3.0)    # capped at 3 days
    t["tsl_fs"] = last
    return t.sort_index()


EV = ["feed_substrate_d_t", "feed_b_d_t", "tsl_fs"]


def time_scaler(cols, w):
    scale = np.array([w if c == "t" else 1.0 for c in cols])
    return FunctionTransformer(lambda X, s=scale: X * s)


def svr_grid(cols, w):
    return GridSearchCV(make_pipeline(StandardScaler(), time_scaler(cols, w), SVR(kernel="rbf")),
                        {"svr__C": [0.3, 1, 3], "svr__gamma": [0.003, 0.01, 0.03], "svr__epsilon": [0.01, 0.03]},
                        cv=GroupKFold(5), scoring="neg_mean_squared_error")


def krr_grid(cols, w):
    return GridSearchCV(make_pipeline(StandardScaler(), time_scaler(cols, w), KernelRidge(kernel="rbf")),
                        {"kernelridge__alpha": [0.01, 0.1, 1.0], "kernelridge__gamma": [0.003, 0.01, 0.03, 0.1]},
                        cv=GroupKFold(5), scoring="neg_mean_squared_error")


class VaryingCoefRidge:
    """log-deviation ~ sum_k B_k(t) * (a_k + x . b_k): ridge on features x spline basis of time."""

    def __init__(self, cols, n_basis):
        self.cols, self.n_basis = cols, n_basis

    def _aug(self, X):
        Xs = self.sc.transform(X[self.cols])
        B = self.spl.transform(X[["t"]])
        return np.hstack([B] + [Xs[:, [k]] * B for k in range(Xs.shape[1])])

    def fit(self, X, y, groups):
        self.sc = StandardScaler().fit(X[self.cols])
        self.spl = SplineTransformer(n_knots=self.n_basis - 1, degree=2, include_bias=True).fit(X[["t"]])
        A = self._aug(X)
        self.m = GridSearchCV(Ridge(), {"alpha": np.logspace(-1, 4, 16)}, cv=GroupKFold(5),
                              scoring="neg_mean_squared_error").fit(A, y, groups=groups)
        return self

    def predict(self, X):
        return self.m.predict(self._aug(X))


def pooled(tr, te, v, cols, make):
    mu = np.log(tr[v]).groupby(tr.j).mean()
    y = np.log(tr[v]) - tr.j.map(mu)
    m = make()
    m.fit(tr[cols], y, groups=tr.batch_id)
    return np.exp(te.j.map(mu).to_numpy() + m.predict(te[cols]))


def per_time_svr(tr, te, v, cols):
    out = pd.Series(np.nan, index=te.index)
    for j, g in tr.groupby("j"):
        y = np.log(g[v]); mu = y.mean()
        m = GridSearchCV(make_pipeline(StandardScaler(), SVR(kernel="rbf")),
                         {"svr__C": [0.3, 1, 3], "svr__gamma": [0.003, 0.01, 0.03], "svr__epsilon": [0.01, 0.03]},
                         cv=5, scoring="neg_mean_squared_error").fit(g[cols], y - mu)
        gt = te[te.j == j]
        out.loc[gt.index] = np.exp(mu + m.predict(gt[cols]))
    return out.to_numpy()


def ridge_model():
    return R.PerTimeRegression(R.FEATURES_BY_TARGET, {v: R.scaled_ridge(k) for v, k in R.ALPHA_SCALE.items()},
                               n_eigencurves=R.N_EIGENCURVES)


def residual_stacking(tab, tr_ids, te_ids, cols):
    """Per-time ridge + pooled SVR on its inner out-of-fold residuals (log scale)."""
    srt = lambda d: d.sort_values(["batch_id", "j"]).reset_index(drop=True)
    inner = []
    for f in E.make_folds(tr_ids, 5, 123):
        inner.append(ridge_model().fit(tab, [b for b in tr_ids if b not in f]).predict(tab, f))
    inner = srt(pd.concat(inner))
    tr = srt(tab[tab.batch_id.isin(tr_ids)])
    full = srt(ridge_model().fit(tab, tr_ids).predict(tab, te_ids))
    te = srt(tab[tab.batch_id.isin(te_ids)])
    out = te[["batch_id", "j", "elapsed_time"]].copy()
    for v in R.TARGETS:
        r = np.log(tr[v]) - np.log(inner[v])
        m = svr_grid(cols[v], 1.0).fit(tr[cols[v]], r, groups=tr.batch_id)
        out[v] = np.exp(np.log(full[v]) + m.predict(te[cols[v]]))
    return out


def main():
    s = int(sys.argv[1])
    raw = D.load_raw()
    TRAIN = D.train_batches(raw)
    tab = add_event_features(R.build_table(raw), raw)
    sel = {v: sorted(set(R.FEATURES_BY_TARGET[v]) | {"t"}) for v in R.TARGETS}
    ev = {v: sel[v] + EV for v in R.TARGETS}
    variants = {}
    for fs_name, fs in (("sel", sel), ("ev", ev)):
        for w in (2, 4):
            variants[f"svr_tw{w}_{fs_name}"] = ("pooled", fs, lambda c, w=w: (lambda: svr_grid(c, w)))
        variants[f"krr_tw3_{fs_name}"] = ("pooled", fs, lambda c: (lambda: krr_grid(c, 3)))
        for K in (4, 6):
            variants[f"vcr_k{K}_{fs_name}"] = ("pooled", fs, lambda c, K=K: (lambda: VaryingCoefRidge(c, K)))
    variants["svr_tw1_ev"] = ("pooled", ev, lambda c: (lambda: svr_grid(c, 1)))
    variants["pt_svr_sel"] = ("pt_svr", sel, None)
    variants["rs_svr_sel"] = ("rs", sel, None)

    rows, t0 = [], time.time()
    for i, fo in enumerate(E.make_folds(TRAIN, 7, s)):
        tr_ids = [b for b in TRAIN if b not in fo]
        tr, te = tab[tab.batch_id.isin(tr_ids)], tab[tab.batch_id.isin(fo)]
        for name, (kind, fs, mk) in variants.items():
            if kind == "rs":
                out = residual_stacking(tab, tr_ids, fo, fs)
            else:
                out = te[["batch_id", "j", "elapsed_time"]].copy()
                for v in R.TARGETS:
                    out[v] = pooled(tr, te, v, fs[v], mk(fs[v])) if kind == "pooled" else per_time_svr(tr, te, v, fs[v])
            rows.append(out.assign(model=name, fold=i, seed=s))
        print(f"seed {s} fold {i} {time.time() - t0:.0f}s", flush=True)
    pd.concat(rows).to_csv(OUT / f"wave2_oof_s{s}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()

"""Optimisation of the hidden-base-volume features (src/base_volume.py), same folds as everything else.

Fast proxies: byproduct = the note's per-day Huber model (src/byproduct_base.py), substrate = our per-time ridge.
Each variant adds base-derived features; choice on CV seeds 0-2, confirmation on fresh seeds 3-5.
Usage (repo root): python src/experiments/base_opt.py [features|model]
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import base_volume as BV
import byproduct_base as BP
import data as D
import evaluation as E
import regression as R

raw = D.load_raw()
TRAIN = D.train_batches(raw)
H = BV.hidden_volumes(raw)
samp = D.measurements(raw, include_to_predict=True)
samp["j"] = samp.groupby("batch_id").cumcount()
DT = samp.pivot(index="batch_id", columns="j", values="elapsed_time").diff(axis=1)[list(range(1, 14))]
m = D.measurements(raw); m["j"] = m.groupby("batch_id").cumcount()
truth = m.set_index(["batch_id", "j"])
LATE = list(range(6, 14))


def base_variants() -> pd.DataFrame:
    """batch_id x candidate base features (static)."""
    s = lambda ks, M=H: M[list(ks)].sum(axis=1)
    # evaporation / withdrawal model fitted on the late intervals (no base addition there): hidden ~ a + b dt
    # (uses the volumes and feed times only, so it is available for test batches; fitted on all batches = inputs only)
    x, y = DT[LATE].to_numpy().ravel(), H[LATE].to_numpy().ravel()
    b, a = np.polyfit(x, y, 1)
    Hc = H - (a + b * DT)                                   # base added per interval, corrected for dt
    late_mean = H[LATE].mean(axis=1)                         # per-batch withdrawal / evaporation level
    v = pd.DataFrame(index=H.index)
    v["b345"] = s((3, 4, 5)); v["b2345"] = s((2, 3, 4, 5)); v["b12345"] = s((1, 2, 3, 4, 5)); v["b3456"] = s((3, 4, 5, 6))
    v["b34"] = s((3, 4)); v["h3"] = H[3]; v["h4"] = H[4]; v["h5"] = H[5]; v["h2"] = H[2]; v["b12"] = s((1, 2))
    v["c345"] = s((3, 4, 5), Hc); v["c2345"] = s((2, 3, 4, 5), Hc); v["c12345"] = s((1, 2, 3, 4, 5), Hc)
    v["r345"] = v.b345 - 3 * late_mean                       # relative to the batch's own late baseline
    v["late_mean"] = late_mean
    v["blate"] = s((11, 12, 13)); v["clate"] = s((11, 12, 13), Hc)
    print(f"evaporation/withdrawal fit on late intervals: hidden = {a:+.4f} {b:+.4f} * dt")
    return v


VARIANTS = {"none": [], "b345": ["b345"], "b2345": ["b2345"], "b12345": ["b12345"], "b3456": ["b3456"], "b34": ["b34"],
            "h3,h4,h5": ["h3", "h4", "h5"], "b345+h2": ["b345", "h2"], "b345+b12": ["b345", "b12"],
            "c345": ["c345"], "c2345": ["c2345"], "c12345": ["c12345"], "r345": ["r345"], "b345+late_mean": ["b345", "late_mean"],
            "b345+blate": ["b345", "blate"], "c345+clate": ["c345", "clate"]}


def huber_byproduct(tab, tr_ids, ids, feats, alpha=0.3, eps=1.35, w_typ=0.2, window=0):
    tr = tab[tab.batch_id.isin(tr_ids)]
    te = tab[tab.batch_id.isin(ids)].copy()
    te["p"] = np.nan
    for j, g in te.groupby("j"):
        gt = tr[(tr.j - j).abs() <= window]
        mu = np.log(gt.byproduct).groupby(gt.j).mean()
        y = np.log(gt.byproduct) - gt.j.map(mu)
        X = gt[feats].assign(dj=gt.j - j) if window else gt[feats]
        Xt = g[feats].assign(dj=0) if window else g[feats]
        mdl = make_pipeline(StandardScaler(), HuberRegressor(alpha=alpha, epsilon=eps, max_iter=2000)).fit(X, y)
        te.loc[g.index, "p"] = np.exp((1 - w_typ) * (mdl.predict(Xt) + mu.loc[j]) + w_typ * mu.loc[j])
    return te.set_index(["batch_id", "j"]).p


def cv(fn, seeds=range(6)):
    out = []
    for s in seeds:
        p = pd.concat([fn([b for b in TRAIN if b not in fo], fo) for fo in E.make_folds(TRAIN, 7, s)])
        out.append(p)
    return out


def mape(p, v):
    y = truth.loc[p.index, v]
    return 100 * np.mean(np.abs(p - y) / y)


def report(name, sc):
    print(f"{name:16s} seeds 0-2 {np.mean(sc[:3]):.3f} | 3-5 {np.mean(sc[3:]):.3f}", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "features"
    tab = BP.build_table(raw)
    v = base_variants()
    for c in v.columns:
        tab[c] = tab.batch_id.map(v[c])
    base_P = [f for f in BP.FEATURES if f != "base_production"]
    if mode == "features":
        print("\n== BYPRODUCT (note's Huber model) ==")
        for name, add in VARIANTS.items():
            sc = [mape(p, "byproduct") for p in cv(lambda tr, te: huber_byproduct(tab, tr, te, base_P + add))]
            report(name, sc)
        print("\n== SUBSTRATE (per-time ridge) ==")
        make = {k: R.scaled_ridge(a) for k, a in R.ALPHA_SCALE.items()}
        for name, add in VARIANTS.items():
            feats = {k: R.FEATURES_BY_TARGET[k] + (add if k == "substrate" else []) for k in R.TARGETS}
            fn = lambda tr, te: R.PerTimeRegression(feats, make, n_eigencurves=R.N_EIGENCURVES).fit(tab, tr).predict(tab, te).set_index(["batch_id", "j"]).substrate
            sc = [mape(p, "substrate") for p in cv(fn)]
            report(name, sc)
    else:                                    # model hyper-parameters for the byproduct Huber model
        feats = base_P + (sys.argv[2].split(",") if len(sys.argv) > 2 else ["b345"])
        print("features:", feats)
        for alpha in (0.1, 0.3, 1.0, 3.0):
            for eps in (1.35, 2.0, 5.0):
                sc = [mape(p, "byproduct") for p in cv(lambda tr, te: huber_byproduct(tab, tr, te, feats, alpha, eps))]
                report(f"a={alpha} eps={eps}", sc)
        for w_typ in (0.0, 0.1, 0.3):
            sc = [mape(p, "byproduct") for p in cv(lambda tr, te: huber_byproduct(tab, tr, te, feats, w_typ=w_typ))]
            report(f"w_typ={w_typ}", sc)
        for window in (1, 2):
            sc = [mape(p, "byproduct") for p in cv(lambda tr, te: huber_byproduct(tab, tr, te, feats, window=window))]
            report(f"window=+-{window}", sc)


if __name__ == "__main__":
    main()

"""Out-of-fold predictions on the SAME folds (7 folds by batch, E.make_folds(train, 7, seed)) for:
    families : the pooled-time non-linear models of src/model_families.py (6 families x 2 feature sets)
    ours     : our hybrid_ensemble (src/run_regression.py: ridge + baselines + hybrid ODE + substrate rule)
    theirs   : the teammate's model (src/v2.py: neighbour-day regression + ODE)
    ours_base / theirs_base : same, with the hidden base volume (src/base_volume.py) added to the substrate and
               byproduct features
    byphuber : standalone byproduct model of the "hidden base volume" note (per-day Huber on log, 80/20 typical)
Usage (repo root):  python src/experiments/families_cv.py {families|ours|theirs|ours_base|theirs_base|byphuber} SEED
Output: results/families/{families|ours|their}_oof_s<SEED>.csv  (then src/experiments/families_analysis.py)
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
import data as D
import evaluation as E
import regression as R

OUT = SRC.parent / "results" / "families"
OUT.mkdir(parents=True, exist_ok=True)
what, s = sys.argv[1], int(sys.argv[2])
raw = D.load_raw()
TRAIN = D.train_batches(raw)
FOLDS = E.make_folds(TRAIN, 7, s)
t0 = time.time()

if what == "families":
    import model_families as F
    TAB = R.build_table(raw)
    rows = []
    for i, fo in enumerate(FOLDS):
        tr_ids = [b for b in TRAIN if b not in fo]
        for fs in ("sel", "wide"):
            for fam in F.FAMILIES:
                rows.append(F.pooled_predict(TAB, tr_ids, fo, fam, fs).assign(model=f"{fam}_{fs}", fold=i, seed=s))
        print(f"seed {s} fold {i} {time.time() - t0:.0f}s", flush=True)
    pd.concat(rows).to_csv(OUT / f"families_oof_s{s}.csv", index=False)

elif what in ("ours", "ours_base", "ours_basec"):
    from run_regression import predict
    TAB = R.build_table(raw)
    feats = dict(R.FEATURES_BY_TARGET)
    if what == "ours_base":
        import base_volume as BV
        TAB = BV.add_base_features(TAB, raw)
        for v in ("substrate", "byproduct"):
            feats[v] = feats[v] + ["base_production"]
    if what == "ours_basec":           # refined base features: substrate 2-5, byproduct 3-5, evaporation-corrected
        import base_volume as BV
        TAB = BV.add_refined_base_features(TAB, raw)
        feats["substrate"] = feats["substrate"] + ["base_c2345"]
        feats["byproduct"] = feats["byproduct"] + ["base_c345"]
    o = pd.concat([predict(raw, TAB, feats, [b for b in TRAIN if b not in fo], fo, 0.2, R.N_EIGENCURVES,
                           ensemble=True, rules=True, ode=True).assign(fold=i, seed=s) for i, fo in enumerate(FOLDS)])
    o.to_csv(OUT / f"{what}_oof_s{s}.csv", index=False)

elif what in ("theirs", "theirs_base"):
    import v2 as V                     # teammate's module (its main() is guarded by __name__)
    d = pd.read_csv(V.DATA_PATH)
    tab = V.build_table(d)
    if what == "theirs_base":          # the note's recipe: base_production as an extra substrate / byproduct input
        import base_volume as BV
        tab["base_production"] = tab.batch_id.map(BV.base_production(raw))
        for v in ("substrate", "byproduct"):
            V.FEAT[v] = V.FEAT[v] + ["base_production"]
    first_topup = d[d.feed_substrate.notna()].groupby("batch_id").elapsed_time.min()
    rows = []
    for i, fo in enumerate(FOLDS):
        tr_ids = [b for b in TRAIN if b not in fo]
        tr, te = tab[tab.batch_id.isin(tr_ids)], tab[tab.batch_id.isin(fo)]
        reg, base = V.regression_predict(tr, te), V.baseline_predict(tr, te)
        st = te[["batch_id", "elapsed_time", "j"]].copy()
        for v in V.T:
            st[v + "_stat"] = (1 - V.W_BASELINE) * reg[v] + V.W_BASELINE * base[v]
        prep = V.prepare(d, tr_ids)
        p = {k: v[0] for k, v in V.PARAMS.items()}
        for name, free, use, nf in [("cells", ["mu_max", "KA", "nA", "kdA", "KdA", "tlag", "kd0", "kd1", "Kd", "klys"], (0, 1), 40),
                                    ("substrate", ["m", "Yxs", "c_B", "c_A", "q_ov", "Kov", "n_ov", "Kx"], (2, 3), 25),
                                    ("byproduct", ["Yp", "q_c", "Kc"], (3,), 30), ("joint", V.NAMES, (0, 1, 2, 3), 30)]:
            p = V.fit(prep, p, free, use, nf)             # same staged fit as in v2.py, on the training fold only
        prep_v = V.prepare(d, fo)
        mech = pd.DataFrame(np.log(np.maximum(V.simulate(p, prep_v), 1e-3)), columns=[o + "_mech" for o in V.OBS])
        mech["batch_id"] = [x[4] for x in prep_v["samples"]]
        mech["_t"] = [round(x[1], 6) for x in prep_v["samples"]]
        st["_t"] = st.elapsed_time.round(6)
        st = st.merge(mech, on=["batch_id", "_t"])
        f1 = st.batch_id.map(first_topup)
        st["after_early"] = (f1 < V.EARLY_TOPUP_DAY) & (st.elapsed_time > f1)
        for v in V.T:                                     # their final combination
            w = np.where(st.after_early & (v == "substrate"), V.W_MECH_SUBSTRATE_AFTER_EARLY_TOPUP, V.W_MECH[v])
            st[v] = np.exp((1 - w) * st[v + "_stat"] + w * st[v + "_mech"])
        st["tcd"] = np.maximum(st.tcd, st.vcd)
        rows.append(st.assign(fold=i, seed=s))
        print(f"seed {s} fold {i} {time.time() - t0:.0f}s", flush=True)
    pd.concat(rows).to_csv(OUT / ("their_oof_s%d.csv" % s if what == "theirs" else "theirs_base_oof_s%d.csv" % s), index=False)

elif what == "byphuber":
    import base_volume as BV
    from sklearn.linear_model import HuberRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    TAB = BV.add_base_features(R.build_table(raw), raw)
    f = D.feeds_long(raw)
    TAB["nCs_tot"] = TAB.batch_id.map(f[(f.feed == "feed_c") & (f.amount <= 0.0013)].groupby("batch_id").size()).fillna(0)
    FE = ["log_vcd0", "log_tcd0", "log_byproduct0", "V0", "feed_substrate_tot", "feed_b_tot", "Vt", "feed_b_cum_t",
          "nCs_tot", "base_production"]
    rows = []
    for i, fo in enumerate(FOLDS):
        tr = TAB[TAB.batch_id.isin([b for b in TRAIN if b not in fo])]
        te = TAB[TAB.batch_id.isin(fo)].copy()
        for j, g in te.groupby("j"):
            gt = tr[tr.j == j]
            typ = np.log(gt.byproduct).mean()
            mdl = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.3, epsilon=1.35, max_iter=1000))
            mdl.fit(gt[FE], np.log(gt.byproduct) - typ)
            te.loc[g.index, "byproduct"] = np.exp(0.8 * (mdl.predict(g[FE]) + typ) + 0.2 * typ)
        rows.append(te[["batch_id", "j", "elapsed_time", "byproduct"]].assign(fold=i, seed=s))
    pd.concat(rows).to_csv(OUT / f"byphuber_oof_s{s}.csv", index=False)
print("done", what, s, flush=True)

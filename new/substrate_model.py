"""Substrate model: mass balance + growth kinetics (physics core) inside a per-sample
statistical model (ridge on log-substrate), validated honestly by batch.

Self-contained in new/ (imports substrate_mb.py sitting next to it; reads the CSVs in new/).

Physics core (substrate_mb.simulate): growth kinetics (Monod x hidden nutrient) coupled to the
EXACT substrate mass balance using the measured volume at every sample (incl. the final one):
    S_j (V_j + r) = S_{j-1} V_{j-1} + C_sub FS_j + C_b FB_j - consumed_j
The unexplained volume (hidden base addition / withdrawal) is absorbed by the measured V_j, which
removes the bias of the volume proxy at the end of the process.

Statistical layer: one ridge per sample index j on log(substrate), features = feed profile, volume
profile (incl. hidden-volume read-outs), controller dose classes, t=0 state, and the physics core's
predictions (log S_mech, log X_mech). Log-target ridge ~ relative-error loss = MAPE-aligned.

Usage (repo root, python = /home/renku/work/.venv/bin/python):
    python new/substrate_model.py quick          # 1 seed, ODE params from fitall (dev only)
    python new/substrate_model.py cv --seed 0 --fold 0   # honest OOF for one fold
    python new/substrate_model.py cv --seed 0             # all folds of one seed
    python new/substrate_model.py eval
    python new/substrate_model.py predict        # -> new/submission_substrate_model.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import substrate_mb as M

HERE = Path(__file__).resolve().parent
ALPHAS = np.logspace(-2, 4, 25)
FEEDS = M.FEEDS


# ---------------------------------------------------------------------------
# feature table: one row per (batch, sample j>=1)
# ---------------------------------------------------------------------------
def build_table(raw: pd.DataFrame, mech: pd.DataFrame | None = None) -> pd.DataFrame:
    samp = raw[raw.volume.notna()].copy()
    samp["j"] = samp.groupby("batch_id").cumcount()
    f = raw[raw.row_type == "feed"]
    feeds = f.melt(id_vars=["batch_id", "elapsed_time"], value_vars=FEEDS,
                   var_name="feed", value_name="amount").dropna(subset=["amount"])
    rows = []
    for b, g in samp.groupby("batch_id", sort=False):
        g = g.sort_values("elapsed_time").reset_index(drop=True)
        fb = feeds[feeds.batch_id == b].sort_values("elapsed_time")
        m0 = g.iloc[0]
        static = {
            "log_S0": np.log(m0.substrate), "log_X0": np.log(m0.vcd),
            "log_T0": np.log(m0.tcd), "log_P0": np.log(m0.byproduct), "V0": m0.volume,
            "fs_tot": fb.loc[fb.feed == "feed_substrate", "amount"].sum(),
            "fb_tot": fb.loc[fb.feed == "feed_b", "amount"].sum(),
            "fc_tot": fb.loc[fb.feed == "feed_c", "amount"].sum(),
            "fd_tot": fb.loc[fb.feed == "feed_d", "amount"].sum(),
            "fs_t_first": fb.loc[fb.feed == "feed_substrate", "elapsed_time"].min(),
            "fs_first_amt": fb.loc[fb.feed == "feed_substrate", "amount"].min(),
            "nC_late": int((fb.loc[fb.feed == "feed_c", "elapsed_time"] >= 11).sum()),
        }
        s = fb[fb.feed == "feed_substrate"].sort_values("elapsed_time")
        fs_times = s.elapsed_time.to_numpy()
        fs_amts = s.amount.to_numpy()
        for i in range(1, len(g)):
            t0, t1 = g.elapsed_time[i - 1], g.elapsed_time[i]
            win = fb[(fb.elapsed_time > t0) & (fb.elapsed_time <= t1)]
            wnext = fb[(fb.elapsed_time > t1) & (fb.elapsed_time <= t1 + 1.6)]
            F = win.amount.sum()
            FSj = win.loc[win.feed == "feed_substrate", "amount"].sum()
            # controller dose right after sample j (in next interval): reveals its view of S_j
            FS_next = wnext.loc[wnext.feed == "feed_substrate", "amount"].sum()
            FB_next = wnext.loc[wnext.feed == "feed_b", "amount"].sum()
            cum = fb[fb.elapsed_time <= t1]
            rows.append({
                "batch_id": b, "j": int(g.j[i]), "elapsed_time": t1,
                "substrate": g.substrate[i] if not g.to_predict[i] else np.nan,
                "vcd": g.vcd[i] if not g.to_predict[i] else np.nan,
                "t": t1, "Vt": g.volume[i], "Vprev": g.volume[i - 1],
                "dV": g.volume[i] - g.volume[i - 1], "hidden": g.volume[i] - g.volume[i - 1] - F,
                "FSj": FSj, "FBj": win.loc[win.feed == "feed_b", "amount"].sum(),
                "FCj": win.loc[win.feed == "feed_c", "amount"].sum(),
                "FDj": win.loc[win.feed == "feed_d", "amount"].sum(),
                "FS_next": FS_next, "FB_next": FB_next,
                "FS_cum": cum.loc[cum.feed == "feed_substrate", "amount"].sum(),
                "FB_cum": cum.loc[cum.feed == "feed_b", "amount"].sum(),
                "FC_cum": cum.loc[cum.feed == "feed_c", "amount"].sum(),
                "nFS": int((cum.feed == "feed_substrate").sum()),
                "nFB": int((cum.feed == "feed_b").sum()),
                # dose classes (0.004 vs 0.006 per unit volume = controller threshold on S_j)
                "bigFS_next": float(FS_next / max(g.volume[i], 1e-9) > 0.005) if FS_next > 0 else 0.0,
                **static,
            })
        # cumulative hidden volume + base_345 per batch, attached after
    tab = pd.DataFrame(rows)
    hid = tab.pivot(index="batch_id", columns="j", values="hidden")
    cumh = hid.cumsum(axis=1)
    tab["hidden_cum"] = [cumh.loc[b, j] for b, j in zip(tab.batch_id, tab.j)]
    tab["base_345"] = tab.batch_id.map(hid[[3, 4, 5]].sum(axis=1))
    tab["base_2345"] = tab.batch_id.map(hid[[2, 3, 4, 5]].sum(axis=1))
    tab["FB_cum_per_V"] = tab.FB_cum / tab.Vt
    tab["FS_cum_per_V"] = tab.FS_cum / tab.Vt
    if mech is not None:
        k = lambda d: d.assign(_t=d.elapsed_time.round(6))
        p = k(mech)[["batch_id", "_t", "substrate", "vcd"]]
        p = p.rename(columns={"substrate": "mech_log_S", "vcd": "mech_log_X"})
        p["mech_log_S"] = np.log(np.maximum(p.mech_log_S, 1e-3))
        p["mech_log_X"] = np.log(np.maximum(p.mech_log_X, 1e-3))
        tab = k(tab).merge(p, on=["batch_id", "_t"], how="left").drop(columns="_t")
    return tab


FEAT = ["j", "t", "Vt", "Vprev", "dV", "hidden", "hidden_cum", "base_345", "base_2345",
        "FSj", "FBj", "FCj", "FDj", "FS_next", "FB_next", "FS_cum", "FB_cum", "FC_cum", "nFS",
        "bigFS_next", "log_S0", "log_X0", "log_T0", "log_P0", "V0",
        "fs_tot", "fb_tot", "fc_tot", "fd_tot", "fs_t_first", "fs_first_amt", "nC_late"]
MECH = ["mech_log_S", "mech_log_X"]
# team's REDUCED substrate feature set (src/regression.py) expressed in our table's names
BASE = ["log_S0", "log_T0", "log_P0", "V0", "fs_tot", "fb_tot", "fs_t_first", "fs_first_amt",
        "t", "Vt", "FS_cum", "nFS", "FB_cum", "nFB"]
FEATSETS = {
    "mech_only": MECH,
    "base": BASE,
    "base_mech": BASE + MECH,
    "full": FEAT + MECH,
    "full_nomech": FEAT,
    "base_mb": BASE + ["hidden", "hidden_cum", "base_345", "base_2345", "FSj", "FBj",
                       "FS_next", "FB_next", "bigFS_next", "dV", "Vprev"],
    "base_mb_mech": BASE + ["hidden", "hidden_cum", "base_345", "base_2345", "FSj", "FBj",
                            "FS_next", "FB_next", "bigFS_next", "dV", "Vprev"] + MECH,
    "base_mb2": BASE + ["hidden", "hidden_cum", "base_345", "base_2345", "FSj", "FBj",
                        "FS_next", "FB_next", "bigFS_next", "dV", "Vprev",
                        "FB_cum_per_V", "FS_cum_per_V"],
}


# ---------------------------------------------------------------------------
# controller dose-class constraint (LateDoseRule, self-contained port)
# ---------------------------------------------------------------------------
def late_dose_classes(raw: pd.DataFrame) -> dict:
    """(batch_id, j) -> True if the substrate bolus right after sample j (j in {11,12}) is the
    big one (dose/V > 0.005). The controller gives the big dose iff the measured substrate at
    that sample was below ~0.96 g/L, so the dose class is a hard constraint on S_j."""
    samp = raw[raw.volume.notna()].copy()
    samp["j"] = samp.groupby("batch_id").cumcount()
    f = raw[raw.row_type == "feed"].melt(id_vars=["batch_id", "elapsed_time"], value_vars=FEEDS,
                                         var_name="feed", value_name="amount").dropna(subset=["amount"])
    fs = f[f.feed == "feed_substrate"].sort_values("elapsed_time")
    mm = samp[["batch_id", "elapsed_time", "j", "volume"]].rename(columns={"elapsed_time": "t_s"})
    fa = pd.merge_asof(fs, mm.sort_values("t_s"), left_on="elapsed_time", right_on="t_s",
                       by="batch_id", direction="backward")
    fa = fa[fa.j.isin([11, 12])]
    return {(b, int(j)): bool(v) for b, j, v in zip(fa.batch_id, fa.j, fa.amount / fa.volume > 0.005)}


def fit_clip(raw: pd.DataFrame, train_ids: list[str], q: float = 0.2):
    classes = late_dose_classes(raw)
    meas = raw[raw.row_type == "measurement"][["batch_id", "elapsed_time", "substrate"]].copy()
    meas["j"] = meas.groupby("batch_id").cumcount()
    s = meas.set_index(["batch_id", "j"]).substrate
    tr_cls = {k: v for k, v in classes.items() if k[0] in train_ids}
    cap = float(s.loc[[k for k, v in tr_cls.items() if v]].quantile(1 - q))
    floor = float(s.loc[[k for k, v in tr_cls.items() if not v]].quantile(q))
    return classes, cap, floor


def apply_clip(pred: pd.DataFrame, classes: dict, cap: float, floor: float) -> pd.DataFrame:
    p = pred.copy()
    for i, (b, j) in enumerate(zip(p.batch_id, p.j)):
        if (b, int(j)) in classes:
            k = p.index[i]
            p.loc[k, "substrate"] = (min(p.loc[k, "substrate"], cap) if classes[(b, int(j))]
                                     else max(p.loc[k, "substrate"], floor))
    return p


def fit_predict_table(raw, train_ids, val_ids, mech_params=None, feats=None, learner="perj"):
    """Fit per-j ridge on log(substrate) over train_ids, predict val_ids. Honest: mech params
    must have been fitted on train_ids only."""
    feats = feats or FEATSETS["full"]
    if mech_params is None:
        mech_params, _ = M.fit_params(raw, train_ids, max_nfev=150)
    all_ids = sorted(set(raw.batch_id), key=lambda s: int(s[5:]))
    mech = {}
    p = M.predict(raw, all_ids, mech_params)
    for _, r in p.iterrows():
        mech[(r.batch_id, round(r.elapsed_time, 6))] = (r.substrate, r.vcd)
    tab = build_table(raw)
    tab["mech_log_S"] = [np.log(max(mech[(b, round(t, 6))][0], 1e-3)) for b, t in zip(tab.batch_id, tab.elapsed_time)]
    tab["mech_log_X"] = [np.log(max(mech[(b, round(t, 6))][1], 1e-3)) for b, t in zip(tab.batch_id, tab.elapsed_time)]
    tr = tab[tab.batch_id.isin(train_ids) & tab.substrate.notna()]
    va = tab[tab.batch_id.isin(val_ids)]
    out = []
    if learner == "pooled":
        X = tr[feats].to_numpy(float)
        y = np.log(tr.substrate.to_numpy(float))
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(X, y)
        pred = np.exp(model.predict(va[feats].to_numpy(float)))
        tmp = va[["batch_id", "j", "elapsed_time"]].copy()
        tmp["substrate"] = pred
        out.append(tmp)
    else:
        for j, g in tr.groupby("j"):
            X = g[feats].to_numpy(float)
            y = np.log(g.substrate.to_numpy(float))
            if learner == "hgb":
                from sklearn.ensemble import HistGradientBoostingRegressor
                model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                                      max_leaf_nodes=8, min_samples_leaf=6,
                                                      l2_regularization=1.0, random_state=0)
            else:
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            model.fit(X, y)
            Xv = va[va.j == j][feats].to_numpy(float)
            pred = np.exp(model.predict(Xv))
            tmp = va[va.j == j][["batch_id", "j", "elapsed_time"]].copy()
            tmp["substrate"] = pred
            out.append(tmp)
    return pd.concat(out, ignore_index=True), mech_params


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_quick(args):
    raw = M.load_raw()
    train, test = M.split_ids(raw)
    pj = json.loads((HERE / "params_mb.json").read_text())["params"]
    for fset in ([args.feats] if args.feats != "ALL" else list(FEATSETS)):
        feats = FEATSETS[fset]
        folds = M.make_folds(train, 7, args.seed)
        rows = []
        for f, val in enumerate(folds):
            tr = [b for b in train if b not in val]
            pred, _ = fit_predict_table(raw, tr, val, mech_params=pj, feats=feats, learner=args.learner)
            classes, cap, floor = fit_clip(raw, tr)
            pred = apply_clip(pred, classes, cap, floor)
            rows.append(pred.assign(fold=f))
        oof = pd.concat(rows)
        p, per_j = M.mape_by_j(oof, raw)
        print(f"[{fset}] QUICK seed {args.seed}: MAPE {p.ape.mean():.3f} | per j: "
              + str({int(k): round(v, 2) for k, v in per_j['mean'].items()}), flush=True)


def cmd_cv(args):
    raw = M.load_raw()
    train, test = M.split_ids(raw)
    folds = M.make_folds(train, 7, args.seed)
    outdir = HERE / "cv"
    outdir.mkdir(exist_ok=True)
    todo = range(7) if args.fold is None else [args.fold]
    for f in todo:
        val = folds[f]
        tr = [b for b in train if b not in val]
        pred, params = fit_predict_table(raw, tr, val, feats=FEATSETS[args.feats], learner=args.learner)   # ODE refit on tr only
        classes, cap, floor = fit_clip(raw, tr)
        pred = apply_clip(pred, classes, cap, floor)
        pred.to_csv(outdir / f"model_{args.feats}_oof_s{args.seed}_f{f}.csv", index=False)
        (HERE / "cv" / f"params_s{args.seed}_f{f}.json").write_text(json.dumps(params))
        p, per_j = M.mape_by_j(pred, raw)
        print(f"seed {args.seed} fold {f}: substrate MAPE {p.ape.mean():.3f}", flush=True)
        print(per_j["mean"].round(2).to_string(), flush=True)


def cmd_eval(args):
    raw = M.load_raw()
    rows = []
    for f in sorted(glob.glob(str(HERE / "cv" / (args.pattern or "model_*_oof_s*_f*.csv")))):
        seed = int(f.split("_s")[1].split("_")[0])
        oof = pd.read_csv(f)
        p, per_j = M.mape_by_j(oof, raw)
        for j, r in per_j.iterrows():
            rows.append(dict(seed=seed, j=j, mape=r["mean"]))
        rows.append(dict(seed=seed, j="ALL", mape=p.ape.mean()))
    df = pd.DataFrame(rows)
    piv = df.pivot(index="j", columns="seed", values="mape")
    piv["mean_all"] = piv.mean(axis=1)
    if all(s in piv.columns for s in (0, 1, 2)):
        piv["mean_0-2"] = piv[[0, 1, 2]].mean(axis=1)
    if all(s in piv.columns for s in (3, 4, 5)):
        piv["mean_3-5"] = piv[[3, 4, 5]].mean(axis=1)
    print(piv.round(3).to_string())


def cmd_predict(args):
    raw = M.load_raw()
    train, test = M.split_ids(raw)
    params, _ = M.fit_params(raw, train, max_nfev=300)
    (HERE / "params_model.json").write_text(json.dumps(params))
    pred, _ = fit_predict_table(raw, train, test, mech_params=params)
    classes, cap, floor = fit_clip(raw, train)
    pred = apply_clip(pred, classes, cap, floor)
    tpl = pd.read_csv(HERE / "bioreaction_template.csv", dtype={"batch_id": str})
    k = lambda d: d.assign(_t=d.elapsed_time.round(6))
    out = k(tpl).merge(k(pred)[["batch_id", "_t", "substrate"]], on=["batch_id", "_t"], how="left")
    assert out.substrate.notna().all() and (out.substrate > 0).all()
    sub = pd.DataFrame({"batch_id": out.batch_id, "elapsed_time": out.elapsed_time,
                        "substrate": out.substrate})
    sub.to_csv(HERE / "submission_substrate_model.csv", index=False)
    print("wrote new/submission_substrate_model.csv", sub.shape)
    print(sub.groupby("batch_id").substrate.apply(lambda s: s.round(3).tolist()).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("quick", "cv", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--fold", type=int, default=None)
        p.add_argument("--feats", type=str, default="full")
        p.add_argument("--learner", type=str, default="perj", choices=["perj","pooled","hgb"])
    p = sub.add_parser("eval")
    p.add_argument("--pattern", type=str, default=None)
    args = ap.parse_args()
    {"quick": cmd_quick, "cv": cmd_cv, "eval": cmd_eval, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()

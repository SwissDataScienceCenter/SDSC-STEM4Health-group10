"""Mean + physics (ODE) + batch-personality hybrid, per target t and per sample day d:

    y_hat_t(b,d) = mu_{t,d} + lambda_{t,d} * (sim_t(b,d) - mu_{t,d}) + w_{t,d}^T x_b

  * mu_{t,d}     : median of the training batches' y_t(.,d) -- robust per-day baseline
                   (MAPE is relative, the median shrugs off the odd weird batch).
  * sim_t(b,d)   : ODE simulator (substrate_mb extended with a 5th state P = byproduct,
                   dP = Yps * qS * Xv), run forward from the t=0 state + the batch's feed
                   plan + the measured volume profile (volume is given for the test too).
                   Used as a FEATURE (its deviation from mu), not as a prediction.
                   Global params are fitted on the training batches only.
  * lambda_{t,d} : shrinkage in [0,1], chosen per (target, day) by inner grouped CV on the
                   training batches over a grid 0..1 (near 0 where noise dominates, near 1
                   where physics is confident).
  * w_{t,d}      : ridge (StandardScaler + RidgeCV) on batch-level features x_b
                   = [log day-0 vcd/tcd/substrate/byproduct, feed_substrate/b/c/d totals,
                      volume added], fitted per day across batches on the residuals
                   y - mu - lambda*(sim - mu).

Honest evaluation: 7 folds by batch x 6 seeds (repo protocol, same folds as
results/families/*_oof_s{0..5}.csv). The ODE is refitted on the training part of each
outer fold only. Alongside the full model, three variants are written for comparison:
  hyb_med   = pure median            (lambda=0, w=0)
  hyb_ridge = median + ridge         (lambda=0)
  hyb_sim   = pure ODE               (lambda=1, w=0)
  hyb_full  = median + lambda*ODE + ridge

Usage (repo root, python = /home/renku/work/.venv/bin/python, PYTHONDONTWRITEBYTECODE=1):
    python new/hybrid_mean.py quick --seed 0                  # 1 seed, reduced ODE budget
    python new/hybrid_mean.py cv --seed 0 [--fold 0]          # -> new/cv/hyb_*_s{seed}_f{f}.csv
    python new/hybrid_mean.py eval                            # MAPE per target / seed group
    python new/hybrid_mean.py predict                         # -> new/submission_hybrid.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import substrate_mb as M

HERE = Path(__file__).resolve().parent
FEEDS = M.FEEDS
TARGETS = M.TARGETS
LAM_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ALPHAS = np.logspace(-2, 3, 13)
VARIANTS = ["med", "ridge", "sim", "full"]

# batch-level feature sets (x_b). x9 = the full list from the spec: 4 day-0 targets +
# 4 feed totals + volume added; x7 drops feed_c / feed_d totals (weakest signals).
XFEATSETS = {
    "x9": ["log_vcd0", "log_tcd0", "log_S0", "log_P0",
           "fs_tot", "fb_tot", "fc_tot", "fd_tot", "V_added"],
    "x7": ["log_vcd0", "log_tcd0", "log_S0", "log_P0",
           "fs_tot", "fb_tot", "V_added"],
}

# ---------------------------------------------------------------------------
# 5-state ODE: substrate_mb + byproduct state P (dP = Yps * qS * Xv)
# ---------------------------------------------------------------------------
N5 = 5
SPEC5 = dict(M.SPEC)
SPEC5["Yps"] = (0.4, 0.01, 2.0)   # g byproduct produced per g substrate consumed


class Grid5(M.Grid):
    def __init__(self, raw, batch_ids, dt=M.DT):
        super().__init__(raw, batch_ids, dt)
        samp = raw[raw.batch_id.isin(batch_ids) & raw.volume.notna()]
        m0 = samp[samp.elapsed_time == 0].set_index("batch_id").loc[self.ids]
        self.x0 = np.column_stack([m0.vcd, m0.tcd - m0.vcd, m0.substrate,
                                   np.ones(len(self.ids)), m0.byproduct]).astype(float)


def rates5(x, p):
    Xv, Xd, S, N = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    Sp = np.maximum(S, 0.0)
    Np = np.maximum(N, 0.0)
    fS = Sp / (Sp + M.EPS_S)
    mu = p["mu_max"] * Sp / (p["Ks"] + Sp) * Np / (p["K_N"] + Np)
    Sn = Sp ** M.FIXED["n_ovf"]
    q_ovf = p["q_ovf"] * Sn / (p["K_ovf"] ** M.FIXED["n_ovf"] + Sn)
    qS = mu / p["Yxs"] + (p["m"] + q_ovf) * fS
    dXv = (mu - p["kd0"]) * Xv
    dXd = p["kd0"] * Xv - p["klys"] * Xd
    dS = -qS * Xv
    dN = -mu / p["Yxn"] * Xv
    dP = p["Yps"] * qS * Xv          # byproduct follows substrate uptake
    return np.stack([dXv, dXd, dS, dN, dP], axis=1)


def feed_conc5(p):
    z = np.zeros(N5)
    out = {f: z.copy() for f in FEEDS}
    out["feed_substrate"][2] = p["C_sub"]
    out["feed_b"][2] = p["C_b"]
    out["feed_b"][3] = p["C_Nb"]
    return out


def simulate5(g: Grid5, p: dict) -> pd.DataFrame:
    dt = g.dt
    x = g.x0.copy()
    V = g.v0.copy()
    conc = feed_conc5(p)
    r = p["r"]
    post = np.empty((g.n_steps + 1, x.shape[0], N5))
    post[0] = x
    for k in range(g.n_steps):
        k1 = rates5(x, p)
        k2 = rates5(x + 0.5 * dt * k1, p)
        k3 = rates5(x + 0.5 * dt * k2, p)
        k4 = rates5(x + dt * k3, p)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        h = g.hidden[k]
        if h.any():
            m = h > 0
            x[m] = x[m] * (V[m, None] / (V[m] + h[m])[:, None])
            V = V + h
        dV = np.zeros_like(V)
        added = np.zeros_like(x)
        for f in FEEDS:
            dv = g.feed_vol[f][k]
            if dv.any():
                dV += dv
                added += dv[:, None] * conc[f][None, :]
        if dV.any():
            Vn = V + dV
            x = (x * V[:, None] + added) / Vn[:, None]
            V = Vn
        s = g.samp_V[k]
        m = ~np.isnan(s)
        if m.any():
            x[m] = x[m] * (V[m, None] / (s[m] + r)[:, None])
            V = np.where(m, s, V)
        post[k + 1] = x
    o = g.obs
    val = post[o.k.to_numpy() + 1, o.b.to_numpy(), :]
    out = o[["batch_id", "elapsed_time", "j"]].copy()
    out["vcd"] = val[:, 0]
    out["tcd"] = val[:, 0] + val[:, 1]
    out["substrate"] = val[:, 2]
    out["byproduct"] = val[:, 4]
    return out


def fit_params5(raw, batch_ids, wS=1.0, wX=0.5, wP=0.5, dt=M.DT, max_nfev=150,
                init=None, verbose=0):
    """Least-squares fit of the 5-state model on the given (training) batches,
    log-residuals on substrate (wS), vcd/tcd (wX) and byproduct (wP)."""
    g = Grid5(raw, batch_ids, dt)
    o = g.obs
    log_y = {v: np.log(np.maximum(o[v].to_numpy(float), 1e-9)) for v in TARGETS if v != "tcd"}
    log_y["tcd"] = np.log(np.maximum(o.tcd.to_numpy(float), 1e-9))

    names = list(SPEC5)
    z0 = np.log([(init or {}).get(n, SPEC5[n][0]) for n in names])
    lb = np.log([max(SPEC5[n][1], 1e-12) for n in names])
    ub = np.log([SPEC5[n][2] for n in names])
    z0 = np.clip(z0, lb + 1e-9, ub - 1e-9)

    def resid(z):
        p = dict(zip(names, np.exp(z)))
        pred = simulate5(g, p)
        r = [wS * (np.log(np.maximum(pred.substrate.to_numpy(float), 1e-6)) - log_y["substrate"]),
             wX * (np.log(np.maximum(pred.vcd.to_numpy(float), 1e-6)) - log_y["vcd"]),
             wX * (np.log(np.maximum(pred.tcd.to_numpy(float), 1e-6)) - log_y["tcd"]),
             wP * (np.log(np.maximum(pred.byproduct.to_numpy(float), 1e-6)) - log_y["byproduct"])]
        r = np.concatenate(r)
        return np.where(np.isfinite(r), r, 1e3)

    sol = least_squares(resid, z0, bounds=(lb, ub), diff_step=1e-4,
                        max_nfev=max_nfev, verbose=verbose)
    params = dict(zip(names, np.exp(sol.x)))
    info = {"cost": float(sol.cost), "nfev": int(sol.nfev), "success": bool(sol.success)}
    return params, info


def simulate_all(raw, params, dt=M.DT):
    all_ids = sorted(set(raw.batch_id), key=lambda s: int(s[5:]))
    return simulate5(Grid5(raw, all_ids, dt), params)


# ---------------------------------------------------------------------------
# batch-level features x_b and measurement / simulation pivots
# ---------------------------------------------------------------------------
def batch_features(raw) -> pd.DataFrame:
    samp = raw[raw.volume.notna()]
    m0 = samp[samp.elapsed_time == 0].set_index("batch_id")
    mlast = samp.sort_values(["batch_num", "elapsed_time"]).groupby("batch_id").tail(1).set_index("batch_id")
    f = raw[raw.row_type == "feed"].groupby("batch_id")[FEEDS].sum()
    xf = pd.DataFrame({
        "log_vcd0": np.log(m0.vcd), "log_tcd0": np.log(m0.tcd),
        "log_S0": np.log(m0.substrate), "log_P0": np.log(m0.byproduct),
        "fs_tot": f.feed_substrate, "fb_tot": f.feed_b,
        "fc_tot": f.feed_c, "fd_tot": f.feed_d,
        "V_added": mlast.volume - m0.volume,
    })
    return xf


def meas_pivot(raw) -> pd.DataFrame:
    m = raw[raw.row_type == "measurement"].copy()
    m["j"] = m.groupby("batch_id").cumcount()
    return m.pivot(index="batch_id", columns="j", values=TARGETS)


# ---------------------------------------------------------------------------
# the per-(target, day) model: mu + lambda*(sim-mu) + ridge(x_b)
# ---------------------------------------------------------------------------
def run_day_models(sim, yp, xf, feats, train_ids, val_ids, lam_grid=LAM_GRID,
                   inner=5, seed=0):
    """Returns (preds: variant -> DataFrame, lam_df). sim/yp: pivots (batch x (t,j)).
    ODE params must already have been fitted on train_ids only."""
    simp = sim.pivot(index="batch_id", columns="j", values=TARGETS)
    et = sim.set_index(["batch_id", "j"]).elapsed_time
    Xall = xf[feats]
    Xtr = Xall.loc[train_ids].to_numpy(float)
    Xva = Xall.loc[val_ids].to_numpy(float)
    tr_arr = np.array(train_ids)
    gkf = GroupKFold(n_splits=min(inner, len(train_ids)))
    splits = list(gkf.split(Xtr, np.zeros(len(train_ids)), groups=tr_arr))
    acc = {v: {t: [] for t in TARGETS} for v in VARIANTS}
    lam_rows = []
    for t in TARGETS:
        for j in range(1, 14):
            ytr = yp[(t, j)].loc[train_ids].to_numpy(float)
            str_ = simp[(t, j)].loc[train_ids].to_numpy(float)
            mu = float(np.median(ytr))
            sva = simp[(t, j)].loc[val_ids].to_numpy(float)
            # --- inner grouped CV: choose lambda on the training batches only
            best_err, best_lam = np.inf, 0.0
            for lam in lam_grid:
                errs = []
                for itr, iva in splits:
                    r = ytr[itr] - mu - lam * (str_[itr] - mu)
                    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
                    pipe.fit(Xtr[itr], r)
                    ph = mu + lam * (str_[iva] - mu) + pipe.predict(Xtr[iva])
                    errs.append(np.mean(np.abs(ytr[iva] - ph) / ytr[iva]))
                e = float(np.mean(errs))
                if e < best_err:
                    best_err, best_lam = e, lam
            lam = float(best_lam)
            # --- final ridge on residuals, all training batches
            pipe_full = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            pipe_full.fit(Xtr, ytr - mu - lam * (str_ - mu))
            pipe0 = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            pipe0.fit(Xtr, ytr - mu)
            acc["med"][t].append(np.full(len(val_ids), mu))
            acc["ridge"][t].append(mu + pipe0.predict(Xva))
            acc["sim"][t].append(sva.copy())
            acc["full"][t].append(mu + lam * (sva - mu) + pipe_full.predict(Xva))
            lam_rows.append(dict(target=t, j=j, lam=lam, inner_mape=100 * best_err, mu=mu))
    preds = {}
    for v in VARIANTS:
        # acc[v][t] is j-major: [j=1 over val_ids, j=2 over val_ids, ...]
        df = pd.DataFrame({"batch_id": np.tile(val_ids, 13),
                           "j": np.repeat(np.arange(1, 14), len(val_ids))})
        df["elapsed_time"] = [et.loc[(b, j)] for j in range(1, 14) for b in val_ids]
        for t in TARGETS:
            df[t] = np.concatenate(acc[v][t])
        preds[v] = df[["batch_id", "j", "elapsed_time", *TARGETS]]
    return preds, pd.DataFrame(lam_rows)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def mape_per_target(pred: pd.DataFrame, raw: pd.DataFrame):
    m = raw[raw.row_type == "measurement"].copy()
    m["j"] = m.groupby("batch_id").cumcount()
    k = lambda d: d.assign(_t=d.elapsed_time.round(6))
    out = {}
    for t in TARGETS:
        p = k(pred)[["batch_id", "_t", "j", t]].merge(
            k(m)[["batch_id", "_t", t]], on=["batch_id", "_t"], suffixes=("_p", ""))
        p["ape"] = 100 * np.abs(p[f"{t}_p"] - p[t]) / p[t]
        out[t] = (float(p.ape.mean()), p.groupby("j").ape.mean())
    return out


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def run_fold(raw, xf, feats, train_ids, val_ids, max_nfev, wS, wX, wP, seed,
             lam_grid=LAM_GRID, inner=5):
    t0 = time.time()
    params, info = fit_params5(raw, train_ids, wS=wS, wX=wX, wP=wP, max_nfev=max_nfev)
    sim = simulate_all(raw, params)
    yp = meas_pivot(raw)
    preds, lam_df = run_day_models(sim, yp, xf, feats, train_ids, val_ids,
                                   lam_grid=lam_grid, inner=inner, seed=seed)
    dt = time.time() - t0
    return preds, lam_df, params, info, dt


def cmd_quick(args):
    raw = M.load_raw()
    train, test = M.split_ids(raw)
    xf = batch_features(raw)
    feats = XFEATSETS[args.xfeats]
    folds = M.make_folds(train, 7, args.seed)
    agg = {v: {t: [] for t in TARGETS} for v in VARIANTS}
    lam_all = []
    oof = []
    for f, val in enumerate(folds):
        tr = [b for b in train if b not in val]
        preds, lam_df, params, info, dt = run_fold(raw, xf, feats, tr, val,
                                                    args.max_nfev, args.ws, args.wx, args.wp,
                                                    args.seed)
        lam_df["fold"] = f
        lam_all.append(lam_df)
        oof.append(preds["full"])
        scores = mape_per_target(preds["full"], raw)
        for v in VARIANTS:
            sc = mape_per_target(preds[v], raw)
            for t in TARGETS:
                agg[v][t].append(sc[t][0])
        print(f"fold {f}: MAPE full " +
              " ".join(f"{t}={scores[t][0]:.3f}" for t in TARGETS) +
              f" | ODE cost {info['cost']:.1f} nfev {info['nfev']} | {dt:.0f}s", flush=True)
    print(f"\n[quick seed {args.seed}] mean over folds:")
    for v in VARIANTS:
        print(f"  {v:5s}: " + " ".join(f"{t}={np.mean(agg[v][t]):.3f}" for t in TARGETS))
    lam = pd.concat(lam_all).groupby(["target", "j"]).lam.mean().unstack(0)
    print("\nchosen lambda (mean over folds), rows=j, cols=target:")
    print(lam.round(2).to_string())
    print("\nfull model per-j MAPE (pooled OOF):")
    pooled = pd.concat(oof)
    sc = mape_per_target(pooled, raw)
    for t in TARGETS:
        print(f"  {t}: overall {sc[t][0]:.3f} | per j " +
              str({int(k): round(v, 2) for k, v in sc[t][1].items()}))


def cmd_cv(args):
    raw = M.load_raw()
    train, test = M.split_ids(raw)
    xf = batch_features(raw)
    feats = XFEATSETS[args.xfeats]
    folds = M.make_folds(train, 7, args.seed)
    outdir = HERE / "cv"
    outdir.mkdir(exist_ok=True)
    todo = range(7) if args.fold is None else [args.fold]
    for f in todo:
        val = folds[f]
        tr = [b for b in train if b not in val]
        preds, lam_df, params, info, dt = run_fold(raw, xf, feats, tr, val,
                                                    args.max_nfev, args.ws, args.wx, args.wp,
                                                    args.seed)
        for v in VARIANTS:
            preds[v].to_csv(outdir / f"hyb_{v}_s{args.seed}_f{f}.csv", index=False)
        lam_df.to_csv(outdir / f"hyb_lambda_s{args.seed}_f{f}.csv", index=False)
        (outdir / f"hyb_params_s{args.seed}_f{f}.json").write_text(json.dumps(params))
        sc = mape_per_target(preds["full"], raw)
        print(f"seed {args.seed} fold {f}: full MAPE " +
              " ".join(f"{t}={sc[t][0]:.3f}" for t in TARGETS) +
              f" | ODE cost {info['cost']:.1f} nfev {info['nfev']} | {dt:.0f}s", flush=True)


def cmd_eval(args):
    raw = M.load_raw()
    for variant in (args.variants.split(",") if args.variants else VARIANTS):
        rows = []
        for fn in sorted(glob.glob(str(HERE / "cv" / f"hyb_{variant}_s*_f*.csv"))):
            seed = int(fn.split("_s")[1].split("_")[0])
            oof = pd.read_csv(fn)
            sc = mape_per_target(oof, raw)
            for t in TARGETS:
                rows.append(dict(seed=seed, target=t, mape=sc[t][0]))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        piv = df.pivot(index="target", columns="seed", values="mape")
        piv["mean_all"] = piv.mean(axis=1)
        if all(s in piv.columns for s in (0, 1, 2)):
            piv["mean_0-2"] = piv[[0, 1, 2]].mean(axis=1)
        if all(s in piv.columns for s in (3, 4, 5)):
            piv["mean_3-5"] = piv[[3, 4, 5]].mean(axis=1)
        print(f"=== variant {variant} (overall MAPE %, rows=target) ===")
        print(piv.round(3).to_string())
        print()
    # per-j for the full variant
    rows = []
    for fn in sorted(glob.glob(str(HERE / "cv" / "hyb_full_s*_f*.csv"))):
        seed = int(fn.split("_s")[1].split("_")[0])
        oof = pd.read_csv(fn)
        sc = mape_per_target(oof, raw)
        for t in TARGETS:
            for j, v in sc[t][1].items():
                rows.append(dict(seed=seed, target=t, j=int(j), mape=v))
    if rows:
        df = pd.DataFrame(rows)
        piv = df.pivot_table(index=["target", "j"], columns="seed", values="mape")
        piv["m_0-2"] = piv[[c for c in (0, 1, 2) if c in piv.columns]].mean(axis=1)
        piv["m_3-5"] = piv[[c for c in (3, 4, 5) if c in piv.columns]].mean(axis=1)
        print("=== full variant, MAPE per (target, j) ===")
        print(piv.round(2).to_string())
    # chosen lambdas
    lams = []
    for fn in sorted(glob.glob(str(HERE / "cv" / "hyb_lambda_s*_f*.csv"))):
        seed = int(fn.split("_s")[1].split("_")[0])
        lams.append(pd.read_csv(fn).assign(seed=seed))
    if lams:
        lam = pd.concat(lams).groupby(["target", "j"]).lam.agg(["mean", "std"]).reset_index()
        piv = lam.pivot(index="j", columns="target", values="mean")
        print("\n=== chosen lambda (mean over all folds/seeds) ===")
        print(piv.round(2).to_string())


def cmd_predict(args):
    raw = M.load_raw()
    train, test = M.split_ids(raw)
    xf = batch_features(raw)
    feats = XFEATSETS[args.xfeats]
    pj = HERE / "params_mb.json"
    init = json.loads(pj.read_text())["params"] if pj.exists() else None
    params, info = fit_params5(raw, train, wS=args.ws, wX=args.wx, wP=args.wp,
                               max_nfev=args.max_nfev, init=init)
    (HERE / "params_hybrid.json").write_text(json.dumps({"params": params, "info": info}, indent=2))
    sim = simulate_all(raw, params)
    yp = meas_pivot(raw)
    preds, lam_df = run_day_models(sim, yp, xf, feats, train, test, inner=5)
    lam_df.to_csv(HERE / "hybrid_lambda_full.csv", index=False)
    print("chosen lambda (full fit):")
    print(lam_df.pivot(index="j", columns="target", values="lam").round(2).to_string())
    pred = preds["full"].copy()
    for t in TARGETS:
        pred[t] = pred[t].clip(lower=1e-3)
    pred["tcd"] = np.maximum(pred["tcd"], pred["vcd"])
    tpl = pd.read_csv(HERE / "bioreaction_template.csv", dtype={"batch_id": str})
    k = lambda d: d.assign(_t=d.elapsed_time.round(6))
    out = k(tpl).merge(k(pred)[["batch_id", "_t", *TARGETS]], on=["batch_id", "_t"], how="left")
    assert out[TARGETS].notna().all().all() and (out[TARGETS] > 0).all().all() and len(out) == len(tpl)
    out = out[tpl.columns]
    out.to_csv(HERE / "submission_hybrid.csv", index=False)
    print("wrote new/submission_hybrid.csv", out.shape)
    sc = None
    print(out.groupby("batch_id").substrate.apply(lambda s: s.round(3).tolist()).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("quick", "cv", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--fold", type=int, default=None)
        p.add_argument("--xfeats", type=str, default="x9", choices=list(XFEATSETS))
        p.add_argument("--ws", type=float, default=1.0)
        p.add_argument("--wx", type=float, default=0.5)
        p.add_argument("--wp", type=float, default=0.5)
        p.add_argument("--max_nfev", type=int, default=150 if name != "quick" else 60)
    p = sub.add_parser("eval")
    p.add_argument("--variants", type=str, default=None)
    args = ap.parse_args()
    {"quick": cmd_quick, "cv": cmd_cv, "eval": cmd_eval, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()

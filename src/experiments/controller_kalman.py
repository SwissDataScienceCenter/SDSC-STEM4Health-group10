"""Feed B controller as an indirect substrate sensor: Gaussian (Kalman-type) update of the substrate trajectory.

Physics: Feed B (which contains substrate) is dosed right after every sample by a feedback controller that replaces
the substrate consumed the day before and corrects the deviation from a set point. Hence, between consecutive
samples, the dose sizes reveal the substrate GIVEN the previous one:
    S_k = a_k + b_k S_{k-1} + c_k d_k / V_k + e_k d_{k-1} / V_{k-1} + eps_k      (d_k = Feed B dose after sample k)
With the TRUE S_{k-1} this law predicts S_k within 3-5 % on days 4-7 (vs 6-7 % for the per-day models); with S_0
known exactly it anchors the chain.
Method (per validation batch): prior S ~ N(mu, Sigma) with mu = current best prediction and Sigma = per-day error
sd x correlation between days (persistent batch offsets), both estimated on the training batches' out-of-fold
errors; observations = the controller law for k in K_OBS (coefficients and eps sd fitted on the training batches'
TRUE substrate). Posterior mean = mu + Sigma H' (H Sigma H' + R)^-1 (z - H mu), where H encodes S_k - b_k S_{k-1}.
Usage (repo root): python src/experiments/controller_kalman.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import control_rules as C
import data as D
import evaluation as E

OUT = SRC.parent / "results" / "families"
raw = D.load_raw()
TRAIN = D.train_batches(raw)
samp = D.measurements(raw, include_to_predict=True)
samp["j"] = samp.groupby("batch_id").cumcount()
Vm = samp.pivot(index="batch_id", columns="j", values="volume")
S_all = samp.pivot(index="batch_id", columns="j", values="substrate")           # NaN on test rows (j > 0)
f = D.feeds_long(raw)
fb = f[f.feed == "feed_b"].sort_values(["batch_id", "elapsed_time"]).copy()
fb["k"] = fb.groupby("batch_id").cumcount() + 1
DOSE = fb.pivot(index="batch_id", columns="k", values="amount") / Vm[list(range(1, 13))].values   # d_k / V_k
DOSE[13] = 0.0                                                                     # no Feed B after sample 13
# substrate feed (Feed A, concentrated) added in the interval (k-1, k], per volume at sample k
_fa = f[f.feed == "feed_substrate"].copy()
_fa = pd.merge_asof(_fa.sort_values("elapsed_time"), samp[["batch_id", "elapsed_time", "j"]].rename(columns={"elapsed_time": "ts"}).sort_values("ts"),
                    left_on="elapsed_time", right_on="ts", by="batch_id", direction="forward")      # j = first sample after the dose
FA = (_fa.groupby(["batch_id", "j"]).amount.sum().unstack().reindex(index=Vm.index, columns=range(0, 14)).fillna(0.0)) / Vm.values
J = list(range(1, 14))


def fit_controller(tr, ks):
    """Per k: S_k ~ S_{k-1} + d_k/V_k + d_{k-1}/V_{k-1} (k = 1: S_1 ~ S_0 + d_1/V_1) on TRUE training substrate."""
    laws = {}
    for k in ks:
        X = pd.DataFrame({"Sp": S_all.loc[tr, k - 1], "d": DOSE.loc[tr, k]})
        if k > 1:
            X["dp"] = DOSE.loc[tr, k - 1]
        if (FA.loc[tr, k] > 0).sum() >= 3:
            X["fa"] = FA.loc[tr, k]
        X = X.loc[:, X.std() > 0]
        y = S_all.loc[tr, k]
        lr = LinearRegression().fit(X, y)
        laws[k] = (lr, float(np.std(y - lr.predict(X), ddof=X.shape[1] + 1)), list(X.columns))
    return laws


def prior_cov(res_rel, shrink=0.3):
    """Relative error sd per day and correlation between days (shrunk toward identity)."""
    sd = res_rel.std()
    Cm = res_rel.corr().to_numpy()
    Cm = (1 - shrink) * Cm + shrink * np.eye(len(Cm))
    return sd.to_numpy(), Cm


def update(mu, s0, sd_rel, Cm, laws, b, scale_R=1.0):
    """Posterior mean of S_1..S_13 for one batch. mu: (13,) prior means."""
    Sig = np.outer(sd_rel * mu, sd_rel * mu) * Cm
    H, z, Rv = [], [], []
    for k, (lr, sig, cols) in laws.items():
        coef = dict(zip(cols, lr.coef_))
        h = np.zeros(13); h[k - 1] = 1.0
        rhs = (lr.intercept_ + coef.get("d", 0.0) * DOSE.loc[b, k] + (coef.get("dp", 0.0) * DOSE.loc[b, k - 1] if k > 1 else 0.0)
               + coef.get("fa", 0.0) * FA.loc[b, k])
        if k == 1:
            rhs += coef["Sp"] * s0                       # S_0 is known exactly
        else:
            h[k - 2] = -coef["Sp"]
        H.append(h); z.append(rhs); Rv.append((scale_R * sig) ** 2)
    H, z, Rm = np.array(H), np.array(z), np.diag(Rv)
    K = Sig @ H.T @ np.linalg.inv(H @ Sig @ H.T + Rm)
    return mu + K @ (z - H @ mu)


def current_best(s):
    ob = pd.read_csv(OUT / f"ours_basec_oof_s{s}.csv").set_index(["batch_id", "j"])
    th = pd.read_csv(OUT / f"their_oof_s{s}.csv").set_index(["batch_id", "j"]).loc[ob.index]
    return np.exp(0.75 * np.log(ob.substrate) + 0.25 * np.log(th.substrate))


def late_rule(p, s):
    out = p.copy()
    for fo in E.make_folds(TRAIN, 7, s):
        d = C.LateDoseRule().fit(raw, [b for b in TRAIN if b not in fo]).apply(p.loc[fo].rename("substrate").reset_index())
        out.loc[list(zip(d.batch_id, d.j))] = d.substrate.to_numpy()
    return out


def run(s, ks, shrink=0.3, scale_R=1.0):
    P = current_best(s)
    Pm = P.unstack()[J]
    rel = (Pm - S_all.loc[Pm.index, J]) / S_all.loc[Pm.index, J]                # relative OOF errors (train)
    new = Pm.copy()
    for fo in E.make_folds(TRAIN, 7, s):
        tr = [b for b in TRAIN if b not in fo]
        laws = fit_controller(tr, ks)
        sd, Cm = prior_cov(rel.loc[tr], shrink)
        for b in fo:
            new.loc[b] = update(Pm.loc[b].to_numpy(), S_all.loc[b, 0], sd, Cm, laws, b, scale_R)
    new = new.clip(lower=0.05).stack()
    new.index.names = ["batch_id", "j"]
    return late_rule(P, s), late_rule(new.loc[P.index], s)


def mape(p):
    y = S_all.stack().loc[p.index]
    return 100 * np.mean(np.abs(p - y) / y)


def main():
    configs = {"K=1-10": list(range(1, 11)), "K=2-10": list(range(2, 11)), "K=1-7": list(range(1, 8)), "K=1-12": list(range(1, 13))}
    for name, ks in configs.items():
        for shrink in (0.3,):
            for scale_R in (1.0, 1.5):
                r = [run(s, ks, shrink, scale_R) for s in range(6)]
                base = [mape(a) for a, _ in r]; new = [mape(b) for _, b in r]
                print(f"{name:7s} shrink={shrink} R x{scale_R}: substrate {np.mean(base[:3]):.3f}/{np.mean(base[3:]):.3f} -> "
                      f"{np.mean(new[:3]):.3f}/{np.mean(new[3:]):.3f}  (seeds 0-2 / 3-5)", flush=True)
    a, b = run(0, list(range(1, 11)))
    y = S_all.stack().loc[a.index]
    per_j = pd.DataFrame({"before": (100 * (a - y).abs() / y).groupby(level=1).mean(), "after": (100 * (b - y).abs() / y).groupby(level=1).mean()})
    print("\nper day (seed 0, K=1-10):\n", per_j.round(2).T.to_string())


if __name__ == "__main__":
    main()

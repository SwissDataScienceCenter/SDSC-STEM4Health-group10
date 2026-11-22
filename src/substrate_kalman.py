"""Feed B controller as an indirect substrate sensor (Gaussian / Kalman-type update of a predicted trajectory).

Physics: Feed B (which contains substrate) is dosed right after every sample by a feedback controller that replaces
the substrate consumed the day before and corrects the deviation from a set point. So consecutive doses reveal the
substrate GIVEN the previous sample:
    S_k = a_k + b_k S_{k-1} + c_k d_k / V_k + e_k d_{k-1} / V_{k-1} + eps_k        (d_k: Feed B dose after sample k)
(k = 1: S_1 = a_1 + b_1 S_0 + c_1 d_1 / V_1, with S_0 measured). With the TRUE S_{k-1} this predicts S_k within
3-5 % on days 4-7 (per-day models: 6-7 %), but S_{k-1} is unknown for a test batch, which is why per-day
regressions could not use the doses. The update combines both pieces of information:
    prior   S ~ N(mu, Sigma)     mu = predicted trajectory, Sigma = per-day relative error sd x correlation between
                                 days (persistent batch offsets), from out-of-fold errors of the training batches
    obs     H S = z + eps        the controller laws for k in OBS_DAYS, fitted on the training batches' TRUE substrate
    post    mu + Sigma H' (H Sigma H' + R)^-1 (z - H mu)
CV (src/experiments/controller_kalman.py, 6 seeds; days, shrinkage and noise scale chosen on seeds 0-2): substrate of
blend_base 6.70 / 6.81 -> 6.62 / 6.73 (seeds 0-2 / 3-5), gain on all 6 seeds. Only feeds, volumes and S_0 are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import data as D
from control_rules import WINDOW

OBS_DAYS = [1, 2, 4, 5, 6, 7]
SHRINK = 0.6           # correlation between days shrunk toward identity
NOISE_SCALE = 1.5      # observation noise inflated (laws are estimated on 42-49 batches)
# Tuned version (src/experiments/controller_kalman2.py, chosen on seeds 0-2): days 1, 2, 4-8, 11, shrink 0.4,
# noise x1.5, substrate-feed (Feed A) dose of the interval in the law when >= 3 training batches have one:
# blend_base substrate 6.616 / 6.733 (version above, LB 7.438) -> 6.574 / 6.719.
TUNED = dict(days=[1, 2, 4, 5, 6, 7, 8, 11], shrink=0.4)
# Control-rule constraints as pseudo-observations (src/experiments/controller_kalman3.py, chosen on seeds 0-2):
#   late dose classes (S11 / S12 below cap if big dose, above floor if small): a violated bound becomes an observation
#   S_k = bound (sd TAU_LATE), so the correction propagates to the neighbouring days through the prior correlations:
#   6.574 / 6.719 -> 6.494 / 6.612;  + mid-run bolus (S(j0) = c_trig, S(j0+1) = c_after, window floor, sd TAU_MID):
#   -> 6.468 / 6.554.
TAU_LATE, TAU_MID = 0.03, 0.1
# Re-tuned with the constraints inside (src/experiments/controller_kalman4.py, chosen on seeds 0-2): days + 12,
# noise x1.25, tau_mid 0.05, late-rule quantile 0.1: 6.468 / 6.554 (LB 7.300 version) -> 6.410 / 6.487.
V4 = dict(days=[1, 2, 4, 5, 6, 7, 8, 11, 12], shrink=0.4, noise_scale=1.25, tau_mid=0.05, late_q=0.1)


class ControllerUpdate:
    def __init__(self, raw: pd.DataFrame):
        samp = D.measurements(raw, include_to_predict=True)
        samp["j"] = samp.groupby("batch_id").cumcount()
        V = samp.pivot(index="batch_id", columns="j", values="volume")
        self.S = samp.pivot(index="batch_id", columns="j", values="substrate")          # NaN on test rows
        f = D.feeds_long(raw)
        fb = f[f.feed == "feed_b"].sort_values(["batch_id", "elapsed_time"]).copy()
        fb["k"] = fb.groupby("batch_id").cumcount() + 1
        self.dose = fb.pivot(index="batch_id", columns="k", values="amount") / V[list(range(1, 13))].values
        self.dose[13] = 0.0
        fa = f[f.feed == "feed_substrate"].sort_values("elapsed_time")
        fa = pd.merge_asof(fa, samp[["batch_id", "elapsed_time", "j"]].rename(columns={"elapsed_time": "ts"}).sort_values("ts"),
                           left_on="elapsed_time", right_on="ts", by="batch_id", direction="forward")   # first sample after
        self.fa = fa.groupby(["batch_id", "j"]).amount.sum().unstack().reindex(index=V.index, columns=range(0, 14)).fillna(0.0) / V.values

    def fit(self, train_ids, rel_errors: pd.DataFrame, days=OBS_DAYS, shrink=SHRINK):
        """rel_errors: (batch x j=1..13) relative out-of-fold errors of the prior predictions on training batches."""
        self.laws = {}
        for k in days:
            X = pd.DataFrame({"Sp": self.S.loc[train_ids, k - 1], "d": self.dose.loc[train_ids, k]})
            if k > 1:
                X["dp"] = self.dose.loc[train_ids, k - 1]
            if (self.fa.loc[train_ids, k] > 0).sum() >= 3:
                X["fa"] = self.fa.loc[train_ids, k]
            X = X.loc[:, X.std() > 0]
            y = self.S.loc[train_ids, k]
            lr = LinearRegression().fit(X, y)
            self.laws[k] = (lr, float(np.std(y - lr.predict(X), ddof=X.shape[1] + 1)), list(X.columns))
        self.sd = rel_errors.std().to_numpy()
        self.corr = (1 - shrink) * rel_errors.corr().to_numpy() + shrink * np.eye(rel_errors.shape[1])
        return self

    def update(self, b: str, mu: np.ndarray, noise_scale=NOISE_SCALE, late=None, mid=None, iters=3, tau_mid=TAU_MID) -> np.ndarray:
        """Posterior mean of S_1..S_13 of batch b given prior means mu (13,).
        late: fitted control_rules.LateDoseRule (constraints at samples 11-12), mid: fitted SubstrateControlRule."""
        Sig = np.outer(self.sd * mu, self.sd * mu) * self.corr
        H, z, Rv = [], [], []
        for k, (lr, sig, cols) in self.laws.items():
            coef = dict(zip(cols, lr.coef_))
            h = np.zeros(13); h[k - 1] = 1.0
            rhs = (lr.intercept_ + coef.get("d", 0.0) * self.dose.loc[b, k] + (coef.get("dp", 0.0) * self.dose.loc[b, k - 1] if k > 1 else 0.0)
                   + coef.get("fa", 0.0) * self.fa.loc[b, k])
            if k == 1:
                rhs += coef["Sp"] * self.S.loc[b, 0]                   # S_0 is measured
            else:
                h[k - 2] = -coef["Sp"]
            H.append(h); z.append(rhs); Rv.append((noise_scale * sig) ** 2)

        def solve(H, z, Rv):
            H, z = np.array(H), np.array(z)
            return mu + Sig @ H.T @ np.linalg.inv(H @ Sig @ H.T + np.diag(Rv)) @ (z - H @ mu)

        def obs(k, val, tau):
            h = np.zeros(13); h[k - 1] = 1.0
            return h, val, tau ** 2

        j0 = mid.trig.get(b, np.nan) if mid is not None else np.nan
        if mid is not None and not np.isnan(j0):          # mid-run bolus: S(j0) ~ c_trig, S(j0+1) ~ c_after
            for k, val in ((int(j0), mid.c_trig), (int(j0) + 1, mid.c_after)):
                h, v, r = obs(k, val, tau_mid); H.append(h); z.append(v); Rv.append(r)
        post = solve(H, z, Rv)
        for _ in range(iters if (late is not None or mid is not None) else 0):
            extra = []
            if late is not None:
                for k in (11, 12):
                    if (b, k) in late.big.index:
                        if late.big.loc[(b, k)] and post[k - 1] > late.cap: extra.append(obs(k, late.cap, TAU_LATE))
                        if not late.big.loc[(b, k)] and post[k - 1] < late.floor: extra.append(obs(k, late.floor, TAU_LATE))
            if mid is not None:
                for k in WINDOW:
                    if (np.isnan(j0) or k not in (j0, j0 + 1)) and post[k - 1] < mid.floor:
                        extra.append(obs(k, mid.floor, tau_mid))
            if not extra:
                break
            post = solve(H + [e[0] for e in extra], z + [e[1] for e in extra], Rv + [e[2] for e in extra])
        return np.maximum(post, 0.05)

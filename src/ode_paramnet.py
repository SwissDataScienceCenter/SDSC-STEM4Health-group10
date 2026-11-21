"""Hybrid 'parameter network': ODE v3 with batch-specific parameters predicted from batch features.

Stage 1  per-batch fit: for every training batch, K key parameters of model v3 are re-estimated (log scale,
         log residuals on vcd / tcd / substrate / byproduct = MAPE-aligned); the other parameters are shared
         (global fit). A Gaussian prior keeps per-batch values near the shared ones. All batches are fitted
         in ONE least_squares call with a block-diagonal Jacobian sparsity (fast).
Stage 2  map batch features (t = 0 state, feed totals...) -> per-batch log-parameters (ridge or small MLP).
Stage 3  simulate a new batch with its predicted parameters.

Results (7-fold CV by batch, 6 seeds): with ridge-predicted parameters the ODE alone reaches VCD ~7.0 % MAPE
(v3 with shared parameters: 8.6 %); blended at 20 % with the regression ensemble it improves vcd and tcd by
0.04-0.09 points on every seed. Substrate / byproduct are NOT improved: even with parameters fitted on each
batch itself (10 free parameters), the v3 structure stays at ~17 % / ~15 % on them, so the ODE is only used
for the cell densities. A small MLP for stage 2 overfits badly with 49 batches (MAPE > 70 %): ridge is used.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import data as D
import mechanistic as M

VARY = ["mu_max", "Yxn", "kd0", "m", "q_ovf", "qPc"]     # growth, peak (nutrient), death, uptake, overflow, P consumption
OBS = ["vcd", "tcd", "substrate", "byproduct"]


def fit_per_batch(raw, ids, shared: dict, prior_sd=0.5, res_sd=0.1, max_nfev=60, verbose=0):
    """Returns DataFrame (batch x VARY) of fitted per-batch parameters."""
    bs = M.prepare(raw, ids)
    B, K = len(ids), len(VARY)
    z0 = np.log(np.array([[shared[k]] * B for k in VARY]).T)            # (B, K)
    lb, ub = z0 - np.log(20), z0 + np.log(20)
    y = np.log(bs.obs[OBS].to_numpy(float))                              # (n_obs, 4)
    w = res_sd / prior_sd

    def params(z):
        Z = np.exp(z.reshape(B, K)); p = dict(shared)
        for k, name in enumerate(VARY):
            p[name] = Z[:, k]
        return p

    def resid(z):
        with np.errstate(all="ignore"):
            pred = M.simulate(M.V3, params(z), bs)[OBS].to_numpy(float)
            r = (np.log(np.maximum(pred, 1e-6)) - y).ravel()
        r = np.where(np.isfinite(r), r, 10.0)
        return np.r_[r, w * (z - z0.ravel())]

    # Jacobian sparsity: residuals of batch b depend only on its K parameters (+ prior rows)
    n_obs = len(bs.obs); S = lil_matrix((n_obs * 4 + B * K, B * K), dtype=int)
    for i, b in enumerate(bs.obs["b"].to_numpy()):
        S[4 * i:4 * i + 4, b * K:(b + 1) * K] = 1
    for q in range(B * K):
        S[n_obs * 4 + q, q] = 1
    sol = least_squares(resid, z0.ravel(), bounds=(lb.ravel(), ub.ravel()), jac_sparsity=S, x_scale="jac",
                        diff_step=1e-3, max_nfev=max_nfev, verbose=verbose)
    return pd.DataFrame(np.exp(sol.x.reshape(B, K)), index=ids, columns=VARY), sol


def simulate_with(raw, ids, shared, per_batch: pd.DataFrame):
    p = dict(shared)
    for name in VARY:
        p[name] = per_batch.loc[ids, name].to_numpy(float)
    return M.simulate(M.V3, p, M.prepare(raw, ids))


class ParamNet:
    """Batch features -> log per-batch parameters (one ridge regression per parameter)."""

    def fit(self, X: pd.DataFrame, P: pd.DataFrame):
        self.cols = list(X.columns); Y = np.log(P[VARY].to_numpy(float))
        self.models = [make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X, Y[:, k])
                       for k in range(len(VARY))]
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(np.exp(np.c_[[m.predict(X[self.cols]) for m in self.models]].T), index=X.index, columns=VARY)


def predict_ode(raw, table, train_ids, ids, shared: dict, features: list[str]):
    """Full pipeline: per-batch fit on train_ids -> ridge param-net -> simulate ids. Returns predictions."""
    X = table[table.j == 1].set_index("batch_id")[features]
    P, _ = fit_per_batch(raw, train_ids, shared, max_nfev=40)
    net = ParamNet().fit(X.loc[train_ids], P)
    return simulate_with(raw, ids, shared, net.predict(X.loc[ids]))

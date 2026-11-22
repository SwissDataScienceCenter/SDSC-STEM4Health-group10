"""Curve-parameter model: PCA of log trajectories, one multi-output ridge from batch-level features to scores."""
import numpy as np, pandas as pd
import common as X
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = X.R
raw, table, train_ids, test_ids = X.load()
JS = [3, 6, 9, 12, 13]
TD = ["Vt", "feed_b_cum_t", "feed_substrate_cum_t"]
W = table.pivot(index="batch_id", columns="j", values="t")
BF = {}
def batch_feats(v):
    st = [c for c in R.FEATURES_BY_TARGET[v] if c not in ("t", "Vt", "feed_substrate_cum_t", "feed_substrate_n_t", "feed_b_cum_t", "feed_b_n_t")]
    g1 = table[table.j == 1].set_index("batch_id")[st]
    parts = [g1]
    for c in TD:
        if c in R.FEATURES_BY_TARGET[v] or c in ("Vt","feed_b_cum_t","feed_substrate_cum_t"):
            P = table.pivot(index="batch_id", columns="j", values=c)[JS]
            P.columns = [f"{c}_j{j}" for j in JS]; parts.append(P)
    parts.append(pd.DataFrame({"t_mean": W.mean(axis=1), "t_last": W[13]}))
    return pd.concat(parts, axis=1)
FE = {v: batch_feats(v) for v in X.TARGETS}
ALPHAS = np.logspace(-2, 4, 25)

def curve_log(tr, ids, K):
    out = {}
    for v in X.TARGETS:
        Z = np.log(table[table.batch_id.isin(tr)].pivot(index="batch_id", columns="j", values=v))
        mu = Z.mean().to_numpy(); Zc = Z.to_numpy() - mu
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        M = Vt[:K]; S = Zc @ M.T
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(FE[v].loc[Z.index], S)
        out[v] = pd.DataFrame(mu + m.predict(FE[v].loc[ids]) @ M, index=ids, columns=Z.columns)
    return out

def to_df(logs, ids):
    keys = X._sorted(table[table.batch_id.isin(ids)][["batch_id", "j", "elapsed_time"]])
    for v in X.TARGETS:
        L = logs[v]
        keys[v] = np.exp([L.loc[b, j] for b, j in zip(keys.batch_id, keys.j)])
    return keys

def curve_pca(K):
    return lambda tr, ids: X.geo_blend(to_df(curve_log(tr, ids, K), ids), X.baseline_scaled(tr, ids), 0.2)

def resid(K):
    def f(tr, ids):
        p = R.PerTimeRegression(R.FEATURES_BY_TARGET, R.ridge).fit(table, tr).predict(table, ids)
        pl = {v: np.log(p.pivot(index="batch_id", columns="j", values=v)) for v in X.TARGETS}
        cl = curve_log(tr, ids, K)
        logs = {v: 0.5 * pl[v].loc[ids, cl[v].columns] + 0.5 * cl[v] for v in X.TARGETS}
        return X.geo_blend(to_df(logs, ids), X.baseline_scaled(tr, ids), 0.2)
    return f

if __name__ == "__main__":
    import time; t0 = time.time()
    print("features:", {v: list(FE[v].columns) for v in X.TARGETS})
    res = {}
    for K in (2, 3, 4):
        res[f"curve_pca_K{K}"] = X.report(f"curve_pca_K{K}", X.lobo(curve_pca(K)))
        print(time.time() - t0, flush=True)
    res["curve_resid_K3"] = X.report("curve_resid_K3", X.lobo(resid(3)))
    print(time.time() - t0)

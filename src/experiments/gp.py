"""GP regression per (target, j) on log targets: ridge+GP-residual and pure GP, blended 0.2 with baseline."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as X
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel, DotProduct
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
R = X.R


class GP(BaseEstimator, RegressorMixin):
    def __init__(self, mode="resid"):
        self.mode = mode

    def fit(self, Xa, y):
        Xa = np.asarray(Xa, float); y = np.asarray(y, float)
        self.sc = StandardScaler().fit(Xa); Z = self.sc.transform(Xa); n = Z.shape[1]
        if self.mode == "resid":
            self.rg = RidgeCV(alphas=R.ALPHAS).fit(Z, y); t = y - self.rg.predict(Z)
            k = C(0.1, (1e-3, 1e1)) * RBF(np.full(n, 3.0), (0.3, 1e2)) + WhiteKernel(0.1, (1e-3, 1e1))
            self.gp = GPR(k, normalize_y=True, n_restarts_optimizer=0, random_state=0).fit(Z, t)
        else:
            k = (C(0.1, (1e-3, 1e2)) * DotProduct(1.0, (1e-2, 1e2)) + C(0.1, (1e-3, 1e1)) * RBF(np.full(n, 3.0), (0.3, 1e2))
                 + WhiteKernel(0.1, (1e-3, 1e1)))
            self.gp = GPR(k, normalize_y=True, n_restarts_optimizer=0, random_state=0).fit(Z, y)
        return self

    def predict(self, Xa):
        Z = self.sc.transform(np.asarray(Xa, float))
        p = self.gp.predict(Z)
        return p + self.rg.predict(Z) if self.mode == "resid" else p


def make(mode, blend=0.2):
    def fp(tr, ids):
        _, table, _, _ = X.load()
        m = R.PerTimeRegression(R.FEATURES_BY_TARGET, lambda: GP(mode))
        p = X._sorted(m.fit(table, tr).predict(table, ids))
        p = p[["batch_id", "j", "elapsed_time", *X.TARGETS]]
        return X.geo_blend(p, X.baseline_scaled(tr, ids), blend) if blend else p
    return fp


if __name__ == "__main__":
    raw, table, train, test = X.load()
    if "time" in sys.argv:
        for mode in ("resid", "pure"):
            t0 = time.time()
            for h in train[:2]:
                make(mode)([b for b in train if b != h], [h])
            print(mode, "2 folds", time.time() - t0, flush=True)
        sys.exit()
    ref = X.score(X.lobo(X.ridge_blend))
    for mode in ("resid", "pure"):
        t0 = time.time()
        raw_oof = X.lobo(make(mode, 0))
        X.report(f"gp_{mode}_unblended", raw_oof)
        oof = X.geo_blend(raw_oof, X.lobo(X.baseline_scaled), 0.2)
        s = X.report(f"gp_{mode}", oof)
        print("runtime", time.time() - t0, flush=True)
        if s["mean"] < ref["mean"] - 0.1:
            X.write_submission(f"gp_{mode}", make(mode)(train, test))

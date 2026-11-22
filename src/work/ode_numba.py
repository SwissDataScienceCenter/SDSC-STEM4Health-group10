"""Numba-jitted vectorized mechanistic ODE for the fed-batch CHO challenge.

Mass-form state M = V*[Xv, Xd, S, P] per batch (V known from observations):
    dM_Xv/dt = (mu - kd)*M_Xv
    dM_Xd/dt = kd*M_Xv
    dM_S/dt  = -(mu/Yxs + ms)*M_Xv            (+ feed impulses on M_S)
    dM_P/dt  = (alpha*mu+beta)*M_Xv - qc*M_Xv*P/(KPc+P)  (+ impulses on M_P)
    mu = mu_max * S/(Ks+S) * KiP/(KiP+P) * max(0, 1 - Xv/Kx)
    kd = kd0 + kd1*P/(KP+P)

RK4 on a uniform grid, vectorized over batches, supports per-batch parameter
rows (params2d shape (n_batches, P)). All 21 parameters are positive, fitted
in log space.
"""
import os
import numpy as np
import pandas as pd
from numba import njit

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]

DATA_CANDIDATES = [
    "/home/renku/work/input-data/Project 1 - DTB",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "input-data", "Project 1 - DTB"),
]

DT = 0.0025
T_END = 13.2
NP = 21
PNAMES = ["mu_max", "Ks", "Kx", "KiP", "kd0", "kd1", "KP",
          "Yxs", "ms", "alpha", "beta", "qc", "KPc",
          "g_fs", "g_fb", "g_fc", "g_fd", "h_fs", "h_fb", "h_fc", "h_fd"]
PBOUNDS = np.array([
    (0.05, 3.0), (0.02, 10.0), (5.0, 200.0), (0.05, 50.0),
    (1e-5, 0.5), (1e-5, 2.0), (0.02, 20.0),
    (0.5, 200.0), (1e-5, 2.0), (1e-5, 2.0), (1e-5, 2.0), (1e-5, 5.0), (0.02, 20.0),
    (0.0, 5000.0), (0.0, 5000.0), (0.0, 50000.0), (0.0, 50000.0),
    (0.0, 200.0), (0.0, 200.0), (0.0, 2000.0), (0.0, 2000.0),
])
P0 = np.array([
    0.7, 0.5, 40.0, 5.0, 0.005, 0.08, 1.0,
    8.0, 0.01, 0.05, 0.01, 0.05, 0.5,
    100.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
])


def find_data_dir():
    for c in DATA_CANDIDATES:
        if os.path.exists(os.path.join(c, "bioreaction_remaining.csv")):
            return os.path.abspath(c)
    raise FileNotFoundError(DATA_CANDIDATES)


def load_batches(data_dir=None):
    data_dir = data_dir or find_data_dir()
    df = pd.read_csv(os.path.join(data_dir, "bioreaction_remaining.csv"))
    tpl = pd.read_csv(os.path.join(data_dir, "bioreaction_template.csv"), dtype={"batch_id": str})
    tpl_ids = set(tpl.batch_id)
    B = {}
    for b, g in df.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        o = g[g.vcd.notna()].reset_index(drop=True)
        f = g[g[FEEDS].notna().any(axis=1)]
        v = g[g.volume.notna()]
        y0row = o.iloc[0]
        B[str(b)] = dict(
            hidden=str(b) in tpl_ids,
            t_obs=o.elapsed_time.values.astype(float),
            y_obs=np.column_stack([o.vcd.values, o.tcd.values - o.vcd.values,
                                   o.substrate.values, o.byproduct.values]).astype(float),
            y0=np.array([y0row.vcd, y0row.tcd - y0row.vcd,
                         y0row.substrate, y0row.byproduct], float),
            f_t=f.elapsed_time.values.astype(float),
            f_a=f[FEEDS].fillna(0.0).values.astype(float),
            v_t=v.elapsed_time.values.astype(float),
            v_v=v.volume.values.astype(float),
            t_pred=(tpl[tpl.batch_id == b].sort_values("elapsed_time").elapsed_time.values.astype(float)
                    if str(b) in tpl_ids else None),
        )
    return B, tpl, df


@njit(cache=True)
def _rhs(M, V, out, p):
    # M, out: (n,4); V: (n,); p: (n, NP)
    n = M.shape[0]
    for j in range(n):
        Xv = M[j, 0] / V[j]
        S = M[j, 2] / V[j]
        P = M[j, 3] / V[j]
        mu_max = p[j, 0]; Ks = p[j, 1]; Kx = p[j, 2]; KiP = p[j, 3]
        kd0 = p[j, 4]; kd1 = p[j, 5]; KP = p[j, 6]
        Yxs = p[j, 7]; ms = p[j, 8]; alpha = p[j, 9]; beta = p[j, 10]
        qc = p[j, 11]; KPc = p[j, 12]
        crowding = 1.0 - Xv / Kx
        if crowding < 0.0:
            crowding = 0.0
        mu = mu_max * S / (Ks + S) * KiP / (KiP + P) * crowding
        kd = kd0 + kd1 * P / (KP + P)
        out[j, 0] = (mu - kd) * M[j, 0]
        out[j, 1] = kd * M[j, 0]
        out[j, 2] = -(mu / Yxs + ms) * M[j, 0]
        out[j, 3] = (alpha * mu + beta) * M[j, 0] - qc * M[j, 0] * P / (KPc + P)


@njit(cache=True)
def _simulate(p, M0, V, raw, dt, n_steps):
    """p: (n,NP) natural scale; M0: (n,4); V: (n_steps+1, n); raw: (n_steps, n, 4).
    Returns M_all (n_steps+1, n, 4)."""
    n = M0.shape[0]
    M = M0.copy()
    M_all = np.zeros((n_steps + 1, n, 4))
    M_all[0] = M
    k1 = np.zeros((n, 4)); k2 = np.zeros((n, 4))
    k3 = np.zeros((n, 4)); k4 = np.zeros((n, 4))
    Vk = np.zeros(n); Vh = np.zeros(n)
    for k in range(n_steps):
        # feed impulses at start of step k
        any_imp = False
        for j in range(n):
            s = 0.0
            for c in range(4):
                if raw[k, j, c] != 0.0:
                    s += 1.0
            if s > 0.0:
                any_imp = True
        if any_imp:
            for j in range(n):
                impS = 0.0
                impP = 0.0
                for c in range(4):
                    a = raw[k, j, c]
                    if a != 0.0:
                        impS += p[j, 13 + c] * a
                        impP += p[j, 17 + c] * a
                M[j, 2] += impS
                M[j, 3] += impP
                M_all[k, j, 2] += impS
                M_all[k, j, 3] += impP
        for j in range(n):
            Vk[j] = V[k, j]
            Vh[j] = 0.5 * (V[k, j] + V[k + 1, j])
        _rhs(M, Vk, k1, p)
        for j in range(n):
            for c in range(4):
                M[j, c] = M_all[k, j, c] + 0.5 * dt * k1[j, c]
        _rhs(M, Vh, k2, p)
        for j in range(n):
            for c in range(4):
                M[j, c] = M_all[k, j, c] + 0.5 * dt * k2[j, c]
        _rhs(M, Vh, k3, p)
        for j in range(n):
            for c in range(4):
                M[j, c] = M_all[k, j, c] + dt * k3[j, c]
        _rhs(M, V[k + 1], k4, p)
        for j in range(n):
            for c in range(4):
                M[j, c] = M_all[k, j, c] + (dt / 6.0) * (k1[j, c] + 2 * k2[j, c] + 2 * k3[j, c] + k4[j, c])
            M_all[k + 1, j, 0] = M[j, 0]
            M_all[k + 1, j, 1] = M[j, 1]
            M_all[k + 1, j, 2] = M[j, 2]
            M_all[k + 1, j, 3] = M[j, 3]
    return M_all


class SimSet:
    def __init__(self, batches, ids, dt=DT, t_end=T_END):
        self.dt = float(dt)
        self.n_steps = int(round(t_end / dt))
        self.ids = list(ids)
        n = self.n = len(self.ids)
        t_grid = np.arange(self.n_steps + 1) * self.dt
        self.M0 = np.zeros((n, 4))
        self.V = np.zeros((self.n_steps + 1, n))
        self.raw = np.zeros((self.n_steps, n, 4))
        self.obs_idx = np.zeros((n, 20), dtype=np.int64)   # padded
        self.obs_n = np.zeros(n, dtype=np.int64)
        self.obs_y = np.zeros((n, 20, 4))
        self.pred_idx = np.zeros((n, 20), dtype=np.int64)
        self.pred_n = np.zeros(n, dtype=np.int64)
        for j, b in enumerate(self.ids):
            bd = batches[b]
            self.M0[j] = bd["y0"] * bd["v_v"][0]
            self.V[:, j] = np.interp(t_grid, bd["v_t"], bd["v_v"])
            for tf, fa in zip(bd["f_t"], bd["f_a"]):
                k = int(round(tf / self.dt))
                if 0 <= k < self.n_steps:
                    self.raw[k, j] += fa
            no = len(bd["t_obs"])
            self.obs_idx[j, :no] = np.clip(np.round(bd["t_obs"] / self.dt).astype(int), 0, self.n_steps)
            self.obs_n[j] = no
            self.obs_y[j, :no] = bd["y_obs"]
            tp = bd["t_pred"]
            if tp is not None and len(tp):
                self.pred_n[j] = len(tp)
                self.pred_idx[j, :len(tp)] = np.clip(np.round(tp / self.dt).astype(int), 0, self.n_steps)

    def simulate(self, params2d):
        """params2d: (n, NP) natural scale -> M_all"""
        return _simulate(np.ascontiguousarray(params2d), self.M0, self.V, self.raw,
                         self.dt, self.n_steps)

    def gather(self, M_all, idx_arr, n_arr, maxn):
        """returns list of (m,4) concentration arrays"""
        out = []
        for j in range(self.n):
            m = n_arr[j]
            if m == 0:
                out.append(None)
                continue
            idx = idx_arr[j, :m]
            Mj = M_all[idx, j, :]
            Vj = self.V[idx, j]
            out.append(Mj / Vj[:, None])
        return out

    def obs_preds(self, params2d):
        M_all = self.simulate(params2d)
        return self.gather(M_all, self.obs_idx, self.obs_n, 20)

    def pred_preds(self, params2d):
        M_all = self.simulate(params2d)
        return self.gather(M_all, self.pred_idx, self.pred_n, 20)


def mape_from_preds(preds, sim, skip_first=True):
    """MAPE% per target col over batches."""
    num = np.zeros(4); den = np.zeros(4)
    for j, pr in enumerate(preds):
        if pr is None:
            continue
        yo = sim.obs_y[j, : sim.obs_n[j]]
        if skip_first and len(pr) > 1:
            pr = pr[1:]; yo = yo[1:]
        for t in range(4):
            num[t] += np.abs(pr[:, t] - yo[:, t]).sum()
            den[t] += np.abs(yo[:, t]).sum()
    return 100.0 * num / np.maximum(den, 1e-12)

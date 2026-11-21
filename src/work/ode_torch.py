"""Vectorized mechanistic ODE (fed-batch CHO bioreactor) with torch autograd.

State per batch in MASS form M = V * [Xv, Xd, S, P] (V = volume, known from
observations), which removes the dilution term D = (dV/dt)/V exactly:

    dM_Xv/dt = (mu - kd) * M_Xv
    dM_Xd/dt = kd * M_Xv
    dM_S /dt = -(mu/Yxs + ms) * M_Xv + feed impulses on M_S
    dM_P /dt = (alpha*mu + beta) * M_Xv - qc * M_Xv * P/(KPc + P) + feed impulses on M_P

    mu = mu_max * S/(Ks+S) * KiP/(KiP+P) * max(0, 1 - Xv/Kx)
    kd = kd0 + kd1 * P/(KP + P)

Feed events are impulses: at event time, M_S += sum_j g_j * feed_j and
M_P += sum_j h_j * feed_j with learnable gains. Concentrations are M / V(t);
V(t) is piecewise-linear from observed volume points (available for hidden
batches too). Integration: fixed-step RK4 on a uniform grid, vectorized over
batches; gradients via torch autograd.
"""
import os
import numpy as np
import pandas as pd
import torch

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]

DATA_CANDIDATES = [
    "/home/renku/work/input-data/Project 1 - DTB",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "input-data", "Project 1 - DTB"),
]

DT = 0.0025
T_END = 13.2

PNAMES = ["mu_max", "Ks", "Kx", "KiP", "kd0", "kd1", "KP",
          "Yxs", "ms", "alpha", "beta", "qc", "KPc",
          "g_fs", "g_fb", "g_fc", "g_fd", "h_fs", "h_fb", "h_fc", "h_fd"]
PBOUNDS = {
    "mu_max": (0.05, 3.0), "Ks": (0.02, 10.0), "Kx": (5.0, 200.0),
    "KiP": (0.05, 50.0), "kd0": (1e-5, 0.5), "kd1": (1e-5, 2.0), "KP": (0.02, 20.0),
    "Yxs": (0.5, 200.0), "ms": (1e-5, 2.0), "alpha": (1e-5, 2.0), "beta": (1e-5, 2.0),
    "qc": (1e-5, 5.0), "KPc": (0.02, 20.0),
    "g_fs": (0.0, 5000.0), "g_fb": (0.0, 5000.0), "g_fc": (0.0, 50000.0), "g_fd": (0.0, 50000.0),
    "h_fs": (0.0, 200.0), "h_fb": (0.0, 200.0), "h_fc": (0.0, 2000.0), "h_fd": (0.0, 2000.0),
}
P0 = {
    "mu_max": 0.7, "Ks": 0.5, "Kx": 40.0, "KiP": 5.0,
    "kd0": 0.005, "kd1": 0.08, "KP": 1.0,
    "Yxs": 8.0, "ms": 0.01, "alpha": 0.05, "beta": 0.01,
    "qc": 0.05, "KPc": 0.5,
    "g_fs": 100.0, "g_fb": 20.0, "g_fc": 0.0, "g_fd": 0.0,
    "h_fs": 0.0, "h_fb": 0.0, "h_fc": 0.0, "h_fd": 0.0,
}


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


def log_bounds():
    lo = np.log([max(PBOUNDS[n][0], 1e-12) for n in PNAMES])
    hi = np.log([PBOUNDS[n][1] for n in PNAMES])
    return lo, hi


def init_log():
    return np.log([max(P0[n], 1e-6) for n in PNAMES])


class Sim:
    """Vectorized RK4 simulator over a set of batches (fixed grid)."""

    def __init__(self, batches, ids, dt=DT, t_end=T_END):
        self.dt = float(dt)
        self.n_steps = int(round(t_end / dt))
        self.ids = list(ids)
        self.n = n = len(self.ids)
        t_grid = np.arange(self.n_steps + 1) * self.dt
        self.t_grid = t_grid
        self.M0 = np.zeros((n, 4))
        self.V = np.zeros((self.n_steps + 1, n))
        self.raw = np.zeros((self.n_steps, n, 4))  # feed amounts at step k
        self.obs_idx, self.obs_y, self.pred_idx = [], [], []
        for j, b in enumerate(self.ids):
            bd = batches[b]
            self.M0[j] = bd["y0"] * bd["v_v"][0]
            self.V[:, j] = np.interp(t_grid, bd["v_t"], bd["v_v"])
            for tf, fa in zip(bd["f_t"], bd["f_a"]):
                k = int(round(tf / self.dt))
                if 0 <= k < self.n_steps:
                    self.raw[k, j] += fa
            self.obs_idx.append(np.clip(np.round(bd["t_obs"] / self.dt).astype(int), 0, self.n_steps))
            self.obs_y.append(bd["y_obs"])
            tp = bd["t_pred"]
            self.pred_idx.append(np.clip(np.round(tp / self.dt).astype(int), 0, self.n_steps)
                                 if tp is not None and len(tp) else None)
        self.raw_t = torch.as_tensor(self.raw, dtype=torch.float64)
        self.V_t = torch.as_tensor(self.V, dtype=torch.float64)
        self.M0_t = torch.as_tensor(self.M0, dtype=torch.float64)

    def run(self, logp):
        """logp: torch tensor (len PNAMES,) in LOG space. Returns M_all (steps+1, n, 4), V."""
        p = torch.exp(logp)
        mu_max, Ks, Kx, KiP = p[0], p[1], p[2], p[3]
        kd0, kd1, KP = p[4], p[5], p[6]
        Yxs, ms, alpha, beta, qc, KPc = p[7], p[8], p[9], p[10], p[11], p[12]
        g = p[13:17]
        h = p[17:21]

        dt = self.dt
        raw = self.raw_t
        V = self.V_t
        M = self.M0_t.clone()
        n_steps, n = self.n_steps, self.n
        zero2 = torch.zeros((n, 2), dtype=torch.float64)

        def rhs(M, Vk):
            Xv = M[:, 0] / Vk
            S = M[:, 2] / Vk
            P = M[:, 3] / Vk
            mu = mu_max * S / (Ks + S) * KiP / (KiP + P) * torch.clamp(1.0 - Xv / Kx, min=0.0)
            kd = kd0 + kd1 * P / (KP + P)
            d0 = (mu - kd) * M[:, 0]
            d1 = kd * M[:, 0]
            d2 = -(mu / Yxs + ms) * M[:, 0]
            d3 = (alpha * mu + beta) * M[:, 0] - qc * M[:, 0] * P / (KPc + P)
            return torch.stack([d0, d1, d2, d3], dim=1)

        M_all = torch.zeros((n_steps + 1, n, 4), dtype=torch.float64)
        M_all[0] = M
        for k in range(n_steps):
            rk = raw[k]
            if rk.abs().sum() > 0:
                M = M + torch.cat([zero2, (rk * g).sum(1, keepdim=True),
                                   (rk * h).sum(1, keepdim=True)], dim=1)
            Vk = V[k]
            Vh = 0.5 * (V[k] + V[k + 1])
            k1 = rhs(M, Vk)
            k2 = rhs(M + 0.5 * dt * k1, Vh)
            k3 = rhs(M + 0.5 * dt * k2, Vh)
            k4 = rhs(M + dt * k3, V[k + 1])
            M = M + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            M_all[k + 1] = M
        return M_all, V

    def gather(self, M_all, V, idx_list):
        preds = []
        for j in range(self.n):
            idx = idx_list[j]
            if idx is None:
                preds.append(None)
                continue
            idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
            Mj = M_all[idx_t, j]
            Vj = V[idx_t, j].unsqueeze(1)
            preds.append(Mj / Vj)
        return preds

    def obs_preds(self, logp):
        M_all, V = self.run(logp)
        return self.gather(M_all, V, self.obs_idx)

    def pred_preds(self, logp):
        M_all, V = self.run(logp)
        return self.gather(M_all, V, self.pred_idx)


def rel_residuals(preds, obs_y, eps=1e-8):
    """List of per-batch (n_obs,4) relative residuals."""
    out = []
    for pr, yo in zip(preds, obs_y):
        yt = torch.as_tensor(yo, dtype=torch.float64)
        out.append((pr - yt) / (yt.abs() + eps))
    return out


def mape_per_target(preds, obs_y, targets=(0, 1, 2, 3)):
    """MAPE (%) per target column over all batches/obs."""
    num = {t: 0.0 for t in targets}
    den = {t: 0 for t in targets}
    for pr, yo in zip(preds, obs_y):
        if pr is None:
            continue
        arr = pr.detach().numpy()
        for t in targets:
            num[t] += np.abs(arr[:, t] - yo[:, t]).sum()
            den[t] += np.abs(yo[:, t]).sum()
    return {t: 100.0 * num[t] / max(den[t], 1e-12) for t in targets}

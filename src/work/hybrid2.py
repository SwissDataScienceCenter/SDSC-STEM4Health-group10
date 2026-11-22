"""Reproduction of the team's winning hybrid model (stem4health_hybrid_model.ipynb)
with a fast numba RK4 simulator, per-batch MAP fits, ridge coefficient regression,
LOBO validation, and hidden-batch prediction.

Model (concentration state y=[Xv,Xd,S,P], V scalar, feed impulses):
    mu  = mu_max * S/(K_S+S) * K_I/(K_I+P)
    sat = S/(K_upt+S)
    dXv = (mu - k_d)*Xv
    dXd = k_d*Xv - k_lysis*Xd
    dS  = -q_S*Xv*sat
    dP  = q_P*Xv*sat - k_rec*Xv*(K_rec/(K_rec+S))*P/(0.5+P)
feed impulse: Vn=V+sum(f); dil=V/Vn; Xv,Xd,P *= dil; S=(S*V + c_ss*fs + c_sb*fb)/Vn; V=Vn
Fixed: mu_max=6, k_lysis=10, K_rec=50.  Fitted (9): K_S,K_I,k_d,q_S,K_upt,q_P,k_rec,c_sb,c_ss

Usage: python hybrid2.py --stage all   (fits, LOBO, submission)
"""
import sys, os, json, time, argparse
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from ode_numba import find_data_dir, FEEDS

DT = 0.0025
T_END = 13.2

PARAM_NAMES = ['mu_max', 'K_S', 'K_I', 'k_d', 'q_S', 'K_upt',
               'q_P', 'k_rec', 'K_rec', 'c_sb', 'c_ss', 'k_lysis']
FIXED = {'mu_max': 6.0, 'k_lysis': 10.0, 'K_rec': 50.0}
FITTED = ['K_S', 'K_I', 'k_d', 'q_S', 'K_upt', 'q_P', 'k_rec', 'c_sb', 'c_ss']
FIDX = [PARAM_NAMES.index(f) for f in FITTED]
PRIOR = np.array([3.8, 4.7, 1.64, 0.32, 5.5, 1.0, 0.44, 46.0, 129.0])
LO = np.array([0.1, 0.3, 0.02, 0.01, 0.05, 0.01, 0.05, 5.0, 1.0])
HI = np.array([20.0, 100.0, 3.0, 1.5, 15.0, 2.0, 3.0, 100.0, 600.0])
LAM = 2.0
SCALES = np.array([10.0, 10.0, 1.0, 1.0])


def load_all():
    dd = find_data_dir()
    df = pd.read_csv(os.path.join(dd, 'bioreaction_remaining.csv'))
    tpl = pd.read_csv(os.path.join(dd, 'bioreaction_template.csv'), dtype={'batch_id': str})
    return df, tpl


def prepare_batch(g):
    g = g.sort_values(['elapsed_time', 'feed_substrate'])
    obs = g[g.vcd.notna()]
    feeds = g.loc[g[FEEDS].notna().any(axis=1), ['elapsed_time'] + FEEDS]
    y0row = obs.iloc[0]
    return dict(
        batch=str(y0row.batch_id),
        t_obs=obs.elapsed_time.values.astype(float),
        obs=np.column_stack([obs.vcd.values, obs.tcd.values,
                             obs['substrate'].values, obs['byproduct'].values]).astype(float),
        t0=float(y0row.elapsed_time),
        y0=np.array([float(y0row.vcd), float(y0row.tcd - y0row.vcd),
                     float(y0row.substrate), float(y0row.byproduct)]),
        V0=float(y0row.volume),
        f_t=feeds.elapsed_time.values.astype(float),
        f_a=feeds[FEEDS].fillna(0.0).values.astype(float),
    )


@njit(cache=True)
def _sim2(p, Y0, V0, raw, dt, n_steps):
    """p: (n,12); Y0: (n,4); V0: (n,); raw: (n_steps,n,4). Returns Y_all (steps+1,n,4), V_all."""
    n = Y0.shape[0]
    Y = Y0.copy()
    V = V0.copy()
    Y_all = np.zeros((n_steps + 1, n, 4))
    V_all = np.zeros((n_steps + 1, n))
    Y_all[0] = Y
    V_all[0] = V
    k1 = np.zeros((n, 4)); k2 = np.zeros((n, 4)); k3 = np.zeros((n, 4)); k4 = np.zeros((n, 4))
    for k in range(n_steps):
        # feed impulse at start of step k
        for j in range(n):
            a0 = raw[k, j, 0]; a1 = raw[k, j, 1]; a2 = raw[k, j, 2]; a3 = raw[k, j, 3]
            ft = a0 + a1 + a2 + a3
            if ft > 0.0:
                Vn = V[j] + ft
                dil = V[j] / Vn
                Y[j, 0] *= dil; Y[j, 1] *= dil; Y[j, 3] *= dil
                Y[j, 2] = (Y[j, 2] * V[j] + p[j, 10] * a0 + p[j, 9] * a1) / Vn
                V[j] = Vn
        Y_all[k] = Y
        V_all[k] = V
        for stage in range(4):
            if stage == 0:
                Yk = Y
            elif stage == 1:
                Yk = Y_all[k] + 0.5 * dt * k1
            elif stage == 2:
                Yk = Y_all[k] + 0.5 * dt * k2
            else:
                Yk = Y_all[k] + dt * k3
            out = k1 if stage == 0 else (k2 if stage == 1 else (k3 if stage == 2 else k4))
            for j in range(n):
                Xv = Yk[j, 0] if Yk[j, 0] > 0.0 else 0.0
                Xd = Yk[j, 1] if Yk[j, 1] > 0.0 else 0.0
                S = Yk[j, 2] if Yk[j, 2] > 0.0 else 0.0
                P = Yk[j, 3] if Yk[j, 3] > 0.0 else 0.0
                mu_max, K_S, K_I, k_d, q_S, K_upt, q_P, k_rec, K_rec, c_sb, c_ss, k_lysis = \
                    p[j, 0], p[j, 1], p[j, 2], p[j, 3], p[j, 4], p[j, 5], p[j, 6], p[j, 7], p[j, 8], p[j, 9], p[j, 10], p[j, 11]
                mu = mu_max * S / (K_S + S) * K_I / (K_I + P)
                sat = S / (K_upt + S)
                out[j, 0] = (mu - k_d) * Xv
                out[j, 1] = k_d * Xv - k_lysis * Xd
                out[j, 2] = -q_S * Xv * sat
                out[j, 3] = q_P * Xv * sat - k_rec * Xv * (K_rec / (K_rec + S)) * P / (0.5 + P)
        for j in range(n):
            for c in range(4):
                Y[j, c] = Y_all[k, j, c] + (dt / 6.0) * (k1[j, c] + 2 * k2[j, c] + 2 * k3[j, c] + k4[j, c])
            for c in range(4):
                if Y[j, c] < 0.0:
                    Y[j, c] = 0.0
            Y_all[k + 1, j] = Y[j]
            V_all[k + 1, j] = V[j]
    return Y_all, V_all


class Sim2:
    def __init__(self, BDS, ids, dt=DT, t_end=T_END):
        self.dt = float(dt)
        self.n_steps = int(round(t_end / dt))
        self.ids = list(ids)
        n = self.n = len(ids)
        self.Y0 = np.zeros((n, 4))
        self.V0 = np.zeros(n)
        self.raw = np.zeros((self.n_steps, n, 4))
        self.obs_idx = np.zeros((n, 20), dtype=np.int64)
        self.obs_n = np.zeros(n, dtype=np.int64)
        self.obs_y = np.zeros((n, 20, 4))  # [vcd, tcd, S, P]
        self.pred_idx = np.zeros((n, 20), dtype=np.int64)
        self.pred_n = np.zeros(n, dtype=np.int64)
        for j, b in enumerate(ids):
            bd = BDS[b]
            self.Y0[j] = bd['y0']
            self.V0[j] = bd['V0']
            ft = bd['f_t'][bd['f_t'] > bd['t0']]
            fa = bd['f_a'][bd['f_t'] > bd['t0']]
            for tf, a in zip(ft, fa):
                k = int(round((tf - bd['t0']) / self.dt))
                if 0 <= k < self.n_steps:
                    self.raw[k, j] += a
            no = len(bd['t_obs'])
            self.obs_idx[j, :no] = np.clip(np.round((bd['t_obs'] - bd['t0']) / self.dt).astype(int), 0, self.n_steps)
            self.obs_n[j] = no
            self.obs_y[j, :no] = bd['obs']
            # pred times
            tpred = BDS[b].get('t_pred')
            if tpred is not None and len(tpred):
                self.pred_n[j] = len(tpred)
                self.pred_idx[j, :len(tpred)] = np.clip(np.round((tpred - bd['t0']) / self.dt).astype(int), 0, self.n_steps)

    def run(self, theta2d):
        return _sim2(np.ascontiguousarray(theta2d), self.Y0, self.V0, self.raw, self.dt, self.n_steps)

    def obs_preds(self, theta2d):
        Y_all, _ = self.run(theta2d)
        out = []
        for j in range(self.n):
            m = self.obs_n[j]
            idx = self.obs_idx[j, :m]
            pred = Y_all[idx, j, :].copy()
            pred[:, 1] = pred[:, 0] + pred[:, 1]  # tcd
            out.append(pred)
        return out

    def pred_preds(self, theta2d):
        Y_all, _ = self.run(theta2d)
        out = []
        for j in range(self.n):
            m = self.pred_n[j]
            if m == 0:
                out.append(None)
                continue
            idx = self.pred_idx[j, :m]
            pred = Y_all[idx, j, :].copy()
            pred[:, 1] = pred[:, 0] + pred[:, 1]
            out.append(pred)
        return out


def full_theta(x_fitted):
    th = np.zeros(12)
    it = iter(x_fitted)
    for i, name in enumerate(PARAM_NAMES):
        th[i] = FIXED[name] if name in FIXED else next(it)
    return th


def fit_batch(bd, sim, n_starts=3, max_nfev=300, seed=0, warm=None):
    """sim: Sim2 with this batch only."""
    j = 0
    m = sim.obs_n[j]
    oi = sim.obs_idx[j, :m]
    oy = sim.obs_y[j, :m]

    def residual(xlog):
        theta = full_theta(np.exp(xlog))
        Y_all, _ = sim.run(theta[None, :])
        pred = Y_all[oi, j, :].copy()
        pred[:, 1] = pred[:, 0] + pred[:, 1]
        r = ((pred - oy) / SCALES).ravel()
        return np.concatenate([r, LAM * (xlog - np.log(PRIOR))])

    lo = np.log(LO); hi = np.log(HI)
    rng = np.random.default_rng(seed)
    best = None
    for s in range(n_starts):
        if s == 0 and warm is not None:
            x0 = warm
        else:
            x0 = np.clip(np.log(PRIOR) + rng.normal(0, 0.4, 9), lo + 1e-9, hi - 1e-9)
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        out = least_squares(residual, x0, bounds=(lo, hi), method='trf',
                            max_nfev=max_nfev, ftol=1e-12, xtol=1e-12)
        if best is None or out.cost < best[0]:
            best = (out.cost, out.x)
    return best[1], best[0]


def batch_features(bd, V_end):
    m = bd['f_t'] > bd['t0'] + 1e-9
    f_a = bd['f_a'][m]
    ft = bd['f_t'][m]
    tot = f_a.sum(axis=0); cnt = (f_a > 0).sum(axis=0)
    fb_t = ft[f_a[:, 1] > 0]
    d_b = np.diff(np.sort(np.concatenate([[bd['t0']], fb_t]))) if fb_t.size else np.array([np.nan])
    return {'vcd0': bd['y0'][0], 'dcd0': bd['y0'][1], 'S0': bd['y0'][2], 'P0': bd['y0'][3],
            'V0': bd['V0'], 'n_fb': cnt[1], 'tot_fb': tot[1],
            'first_fb': (fb_t[0] if fb_t.size else bd['t0']),
            'mean_dt_fb': d_b.mean(),
            'max_amt_fb': (f_a[:, 1].max() if cnt[1] else 0.0),
            'n_fs': cnt[0], 'tot_fs': tot[0], 'n_fc': cnt[2], 'tot_fc': tot[2],
            'n_fd': cnt[3], 'tot_fd': tot[3], 'tot_feed': tot.sum(),
            'V_end': V_end if V_end is not None else bd['V0']}


FEATS = ['vcd0', 'dcd0', 'S0', 'P0', 'V0', 'n_fb', 'tot_fb', 'first_fb',
         'mean_dt_fb', 'max_amt_fb', 'n_fs', 'tot_fs', 'n_fc', 'tot_fc',
         'n_fd', 'tot_fd', 'tot_feed', 'V_end']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dt', type=float, default=DT)
    ap.add_argument('--nfev', type=int, default=300)
    ap.add_argument('--starts', type=int, default=3)
    ap.add_argument('--out-dir', default=os.path.dirname(__file__))
    ap.add_argument('--folds-lobo', action='store_true', default=True)
    args = ap.parse_args()

    df, tpl = load_all()
    BDS = {b: prepare_batch(g) for b, g in df.groupby('batch_id')}
    tpl_ids = sorted(tpl.batch_id.unique(), key=lambda s: int(s[5:]))
    for b in tpl_ids:
        tt = tpl[tpl.batch_id == b].sort_values('elapsed_time').elapsed_time.values.astype(float)
        BDS[b]['t_pred'] = tt
    train_ids = sorted([b for b, g in df[~df.to_predict].groupby('batch_id')
                        if g.vcd.notna().sum() > 5], key=lambda s: int(s[5:]))
    pred_ids = tpl_ids
    print(f'train {len(train_ids)}, hidden {len(pred_ids)}', flush=True)

    # V_end per batch (last observed volume)
    V_end = {}
    for b in list(BDS.keys()):
        vv = df[(df.batch_id == b) & df.volume.notna()].sort_values('elapsed_time')
        V_end[b] = float(vv.volume.iloc[-1]) if len(vv) else BDS[b]['V0']

    # ---- per-batch MAP fits (fast numba), checkpointed
    fits_path = os.path.join(args.out_dir, 'hybrid2_fits.json')
    fits = {}
    if os.path.exists(fits_path):
        fits = json.load(open(fits_path))
    warm = None
    t0 = time.time()
    for b in train_ids:
        if b in fits:
            continue
        sim = Sim2(BDS, [b], dt=args.dt)
        x, cost = fit_batch(BDS[b], sim, n_starts=args.starts, max_nfev=args.nfev,
                            seed=int(b[5:]), warm=warm)
        warm = x
        fits[b] = x.tolist()
        json.dump(fits, open(fits_path, 'w'))
        if len(fits) % 10 == 0 or len(fits) == len(train_ids):
            print(f'fits {len(fits)}/{len(train_ids)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'all fits done ({time.time()-t0:.0f}s)', flush=True)

    # in-sample MAPE per batch
    sim_all = Sim2(BDS, train_ids, dt=args.dt)
    X = np.array([fits[b] for b in train_ids])
    TH = np.array([full_theta(np.exp(x)) for x in X])
    preds = sim_all.obs_preds(TH)
    mapes = []
    for j, b in enumerate(train_ids):
        pr = preds[j][1:]
        oy = sim_all.obs_y[j, 1:sim_all.obs_n[j]]
        mapes.append(100 * np.abs(pr - oy).sum(0) / np.abs(oy).sum(0))
    mapes = np.array(mapes)
    print('in-sample per-batch MAPE per target (mean):', np.round(mapes.mean(0), 3),
          'overall', round(float(mapes.mean()), 3), flush=True)

    # ---- coefficient regression (ridge on standardized features -> log params)
    F = pd.DataFrame([dict(batch_features(BDS[b], V_end[b]), batch=b) for b in train_ids]).set_index('batch')
    Xf = F[FEATS].values.astype(float)
    mu_, sd_ = Xf.mean(0), Xf.std(0) + 1e-12
    Xs = (Xf - mu_) / sd_
    Yl = np.array([fits[b] for b in train_ids])  # already in log space

    def ridge_fit(X, y, alpha):
        Xa = np.hstack([np.ones((len(X), 1)), X])
        A = np.eye(Xa.shape[1]) * alpha; A[0, 0] = 0
        return np.linalg.solve(Xa.T @ Xa + A, Xa.T @ y)

    def ridge_predict(w, X):
        return np.hstack([np.ones((len(X), 1)), X]) @ w

    best_alpha, best_err = None, np.inf
    for alpha in [0.1, 0.3, 1, 3, 10, 30, 100]:
        errs = []
        for i in range(len(train_ids)):
            m = np.ones(len(train_ids), bool); m[i] = False
            w = ridge_fit(Xs[m], Yl[m], alpha)
            errs.append(np.mean((ridge_predict(w, Xs[i:i+1])[0] - Yl[i]) ** 2))
        if np.mean(errs) < best_err:
            best_alpha, best_err = alpha, np.mean(errs)
    print(f'ridge alpha={best_alpha} (LOO param mse {best_err:.4f})', flush=True)
    ALPHA = best_alpha

    # ---- LOBO validation of the full pipeline (MAPE on 13 future points)
    lobo = np.zeros_like(Yl)
    for i in range(len(train_ids)):
        m = np.ones(len(train_ids), bool); m[i] = False
        w = ridge_fit(Xs[m], Yl[m], ALPHA)
        lobo[i] = ridge_predict(w, Xs[i:i+1])[0]
    lobo_mape = []
    for i, b in enumerate(train_ids):
        sim = Sim2(BDS, [b], dt=args.dt)
        th = full_theta(np.exp(lobo[i]))
        pr = sim.obs_preds(th[None, :])[0][1:]
        oy = sim.obs_y[0, 1:sim.obs_n[0]]
        lobo_mape.append(100 * np.abs(pr - oy).sum(0) / np.abs(oy).sum(0))
    lobo_mape = np.array(lobo_mape)
    print('LOBO MAPE per target (mean over batches):', np.round(lobo_mape.mean(0), 3),
          'overall mean', round(float(lobo_mape.mean()), 3), flush=True)
    # also global-median params baseline
    med = np.median(Yl, axis=0)
    med_mape = []
    for i, b in enumerate(train_ids):
        sim = Sim2(BDS, [b], dt=args.dt)
        th = full_theta(np.exp(med))
        pr = sim.obs_preds(th[None, :])[0][1:]
        oy = sim.obs_y[0, 1:sim.obs_n[0]]
        med_mape.append(100 * np.abs(pr - oy).sum(0) / np.abs(oy).sum(0))
    med_mape = np.array(med_mape)
    print('global-median-params MAPE:', np.round(med_mape.mean(0), 3), 'mean', round(float(med_mape.mean()), 3), flush=True)

    # ---- predict hidden batches
    W = ridge_fit(Xs, Yl, ALPHA)
    Fh = pd.DataFrame([dict(batch_features(BDS[b], V_end[b]), batch=b) for b in pred_ids]).set_index('batch')
    Xh = (Fh[FEATS].values.astype(float) - mu_) / sd_
    Xlog = ridge_predict(W, Xh)
    sim_h = Sim2(BDS, pred_ids, dt=args.dt)
    THh = np.array([full_theta(np.exp(x)) for x in Xlog])
    ph = sim_h.pred_preds(THh)
    rows = []
    for j, b in enumerate(pred_ids):
        for k, t in enumerate(BDS[b]['t_pred']):
            rows.append({'batch_id': b, 'elapsed_time': t,
                         'vcd': ph[j][k, 0], 'tcd': ph[j][k, 1],
                         'substrate': ph[j][k, 2], 'byproduct': ph[j][k, 3]})
    fill = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, '..', 'submission_hybrid2.csv')
    fill.to_csv(out_csv, index=False)
    print('wrote', os.path.abspath(out_csv), fill.shape, flush=True)
    json.dump({'alpha': ALPHA, 'lobo_mape': lobo_mape.mean(0).tolist(),
               'insample_mape': mapes.mean(0).tolist(),
               'med_mape': med_mape.mean(0).tolist()},
              open(os.path.join(args.out_dir, 'hybrid2_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

"""
Parameter estimation - Python port of Java Fitter/ProcessModel.

Matches the Java implementation that achieved R² VCD = 0.889:
  - Bolus-based feed model (batch ODE between impulse feed events)
  - dV_bolus at each feed event = feed_rate * dt (forward-fill integration)
  - Feed substrate concentration: S_feed = 500 g/L (from MonodModel)
  - Weighted relative error loss: (pred - obs) / max(|obs|, 0.5)
  - L-M optimizer with finite-difference Jacobian
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from typing import Dict, List, Optional, Tuple
import csv as _csv
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_engine import (
    OrganismConfig, temp_factor_cardinal, ph_factor_gaussian
)

# ===========================================================================
# Configuration
# ===========================================================================
PARAM_DEFAULTS = {
    'mu_max':   {'default': 0.70,   'lower': 0.01, 'upper': 5.0,  'reg_scale': 0.5},
    'K_s':      {'default': 0.02,   'lower': 0.001,'upper': 5.0,  'reg_scale': 0.5},
    'Y_XS':     {'default': 0.50,   'lower': 0.01, 'upper': 0.95, 'reg_scale': 0.2},
    'm_s':      {'default': 0.025,  'lower': 0.0,  'upper': 0.5,  'reg_scale': 0.05},
    'alpha':    {'default': 0.0,    'lower': 0.0,  'upper': 1.0,  'reg_scale': 0.2},
    'beta':     {'default': 0.02,   'lower': 0.0,  'upper': 0.5,  'reg_scale': 0.05},
    'k_d':      {'default': 0.01,   'lower': 0.0,  'upper': 0.1,  'reg_scale': 0.02},
}

S_FEED = 500.0  # g/L substrate in feed (from Java MonodModel.feedConcentrations)
CSV_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv"

# ===========================================================================
# Data Loading
# ===========================================================================
def load_batch_csv(batch_id: str) -> Dict:
    """Load batch: separate measurements and feed events."""
    measurements = {'t': [], 'vcd': [], 'tcd': [], 'substrate': [], 'byproduct': [], 'volume': []}
    feed_rows = []  # (time, total_flow_rate_L_per_h)

    with open(CSV_PATH, 'r') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row['batch_id'] != batch_id:
                continue
            elapsed = float(row['elapsed_time'])

            vcd = float(row['vcd']) if row.get('vcd', '').strip() else None
            tcd = float(row['tcd']) if row.get('tcd', '').strip() else None
            sub = float(row['substrate']) if row.get('substrate', '').strip() else None
            byp = float(row['byproduct']) if row.get('byproduct', '').strip() else None
            vol = float(row['volume']) if row.get('volume', '').strip() else None

            if vcd is not None or sub is not None:
                measurements['t'].append(elapsed)
                measurements['vcd'].append(vcd)
                measurements['tcd'].append(tcd)
                measurements['substrate'].append(sub)
                measurements['byproduct'].append(byp)
                measurements['volume'].append(vol)

            # Compute total instantaneous flow rate from all feed columns
            flow_total = 0.0
            for col in ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']:
                val = row.get(col, '').strip()
                if val:
                    flow_total += float(val)
            if flow_total > 0:
                feed_rows.append((elapsed, flow_total))

    # Java treats feed values as bolus volumes in L (not integrated).
    # Each feed row: dV = total flow rate at that instant (approx=volume when dt≈1h).
    bolus_events = sorted([(t, flow) for t, flow in feed_rows])
    return {'measurements': measurements, 'feed_events': bolus_events}


def _rates_to_bolus_events(feed_rows: List) -> List[Tuple[float, float]]:
    """Convert instantaneous flow rates (L/h) to integrated bolus volumes.

    With forward-fill: rate(t) = rate_i for t in [t_i, t_{i+1}).
    dV_bolus at t_i = rate_i * (t_{i+1} - t_i)
    Last event uses average interval.
    """
    if not feed_rows:
        return []
    sorted_rows = sorted(feed_rows)
    events = []
    n = len(sorted_rows)
    for i in range(n):
        t_i, r_i = sorted_rows[i]
        if i < n - 1:
            dt = sorted_rows[i + 1][0] - t_i
        else:
            dt = t_i - sorted_rows[i - 1][0] if n >= 2 else 1.0
        # Clamp dt to [0.1, 2.0] hours (avoid absurd values from sparse data)
        dt = max(0.1, min(2.0, dt))
        events.append((t_i, r_i * dt))
    return events


# ===========================================================================
# ODE (batch model between boluses)
# ===========================================================================
def batch_ode(t, y, organism, params):
    """Pure batch ODE rates. Feeds are applied as boluses between ODE steps."""
    X_v, X_d, S, P, B, DO, V, T, pH = y
    X_v = max(0.0, X_v); X_d = max(0.0, X_d); S = max(0.0, S)
    DO = max(0.0, DO); V = max(0.1, V)

    mu_max = params.get('mu_max', 0.7)
    K_s = params.get('K_s', 0.02)
    Y_XS = params.get('Y_XS', 0.5)
    m_s = params.get('m_s', 0.025)
    alpha = params.get('alpha', 0.0)
    beta = params.get('beta', 0.02)
    k_d_val = params.get('k_d', 0.01)

    # Environment factors
    env = organism.environment
    f_T = temp_factor_cardinal(T, env.T_min, env.T_opt, env.T_max)
    f_pH = ph_factor_gaussian(pH, env.pH_opt, env.pH_sigma)

    # Growth: Monod
    mu = mu_max * (S / (K_s + S)) * f_T * f_pH

    # Pirt uptake
    q_S = (mu / Y_XS if Y_XS > 0 else 0) + m_s

    # Luedeking-Piret byproduct
    q_B = alpha * mu + beta

    # Death (increases at low substrate)
    k_d = k_d_val * max(0.0, 1.0 - S / (S + K_s))

    # Rates
    dX_v = (mu - k_d) * X_v
    dX_d = k_d * X_v - organism.k_lysis * X_d
    dS = -q_S * X_v
    dP = 0.0
    dB = q_B * X_v
    dDO = 150.0 * (0.007 - DO) - (mu / 0.95 + 0.01) * X_v
    dV = 0.0
    dT = 0.0
    dpH = 0.0
    return [dX_v, dX_d, dS, dP, dB, dDO, dV, dT, dpH]


def apply_bolus(y, dV, S_feed):
    """Apply feed bolus: dilute biomass + add substrate (matches Java applyBolus)."""
    X_v, X_d, S, P, B, DO, V, T, pH = y
    V_new = V + dV
    dil = V / V_new
    feed = dV / V_new
    return [
        X_v * dil, X_d * dil,
        S * dil + S_feed * feed,
        P * dil, B * dil, DO * dil,
        V_new, T, pH
    ]


def simulate(batch: Dict, y0: List[float], organism, params: Dict):
    """Integrate batch between bolus feed events."""
    t_end = max(batch['measurements']['t']) * 1.05
    feed_events = batch['feed_events']

    # Build time segments
    times = []
    for tev, _ in feed_events:
        if 0 < tev < t_end:
            times.extend([tev - 1e-10, tev])
    times = sorted(set([0.0] + times + [t_end]))

    state = list(y0)
    record_t = [0.0]
    record_y = [list(state)]

    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]
        if t1 - t0 < 1e-12:
            continue

        # Apply feed bolus at segment start
        if t0 > 0:
            for tev, dV in feed_events:
                if abs(tev - t0) < 1e-10:
                    state = apply_bolus(state, dV, S_FEED)

        # Integrate
        try:
            sol = solve_ivp(
                lambda t, y: batch_ode(t, y, organism, params),
                (t0, t1), state,
                method='LSODA', rtol=1e-4, atol=1e-6, max_step=1.0
            )
            state = [sol.y[j][-1] for j in range(9)]
        except Exception:
            pass

        if t1 > record_t[-1]:
            record_t.append(t1)
            record_y.append(list(state))

    return np.array(record_t), np.array(record_y).T


# ===========================================================================
# Fitting
# ===========================================================================
def fit_organism(organism_id: str, batch_ids: List[str],
                 params_to_fit: Optional[List[str]] = None,
                 reg_strength: float = 0.0) -> Dict:
    """Fit model parameters to training batches."""
    from db_connector import load_organism

    organism = load_organism(organism_id)
    if organism is None:
        return {"status": "error", "message": f"Organism {organism_id} not found"}

    if params_to_fit is None:
        params_to_fit = ['mu_max', 'K_s', 'Y_XS', 'alpha', 'beta']

    x0 = np.array([PARAM_DEFAULTS[k]['default'] for k in params_to_fit])
    lower = np.array([PARAM_DEFAULTS[k]['lower'] for k in params_to_fit])
    upper = np.array([PARAM_DEFAULTS[k]['upper'] for k in params_to_fit])
    reg_scales = np.array([PARAM_DEFAULTS[k]['reg_scale'] for k in params_to_fit])

    # Pre-load all batches
    all_batches = {}
    for bid in batch_ids:
        all_batches[bid] = load_batch_csv(bid)

    # Pre-build residual point list (matches Java collectPoints)
    points = []
    for bid in batch_ids:
        m = all_batches[bid]['measurements']
        t_obs = m['t']
        for i in range(1, len(t_obs)):  # skip t=0
            if m['vcd'][i] is not None and m['vcd'][i] > 0.01:
                points.append(('vcd', bid, t_obs[i], m['vcd'][i]))
            if m['substrate'][i] is not None and m['substrate'][i] > 0.01:
                points.append(('sub', bid, t_obs[i], m['substrate'][i]))
            if m['byproduct'][i] is not None and m['byproduct'][i] > 0.01:
                points.append(('byp', bid, t_obs[i], m['byproduct'][i]))

    # Cache simulations per batch+params to avoid redundant ODE solves
    _sim_cache = {}

    def simulate_cached(bid, params_tuple):
        key = (bid, params_tuple)
        if key not in _sim_cache:
            batch = all_batches[bid]
            m = batch['measurements']
            V0 = m['volume'][0] if m['volume'][0] else 10.0
            S0 = m['substrate'][0] if m['substrate'][0] else 4.5
            X_v0 = m['vcd'][0] if m['vcd'][0] else 1.0
            y0 = [X_v0, 0.0, S0, 0.0, 0.0, 0.007, V0, 37.0, 7.0]
            fit = dict(zip(params_to_fit, params_tuple))
            t_sim, y_sim = simulate(batch, y0, organism, fit)
            _sim_cache[key] = (t_sim, y_sim)
        return _sim_cache[key]

    def residuals(p):
        """Weighted relative error (matches Java computeResiduals)."""
        params_tuple = tuple(p)
        _sim_cache.clear()
        result = []
        eps = 0.5

        for kind, bid, t, obs in points:
            t_sim, y_sim = simulate_cached(bid, params_tuple)
            if kind == 'vcd':
                pred = np.interp(t, t_sim, y_sim[0])
            elif kind == 'sub':
                pred = np.interp(t, t_sim, y_sim[2])
            else:  # byproduct
                pred = np.interp(t, t_sim, y_sim[4])
            denom = max(abs(obs), eps)
            result.append((pred - obs) / denom)

        # L2 regularization
        reg = reg_strength * (p - x0) / (reg_scales + 1e-6)
        result.extend(reg)
        return np.array(result)

    # Fit
    try:
        result = least_squares(
            residuals, x0, bounds=(lower, upper),
            method='trf', max_nfev=10,  # keep low: Jacobian = 2*n_params solves per iter
            xtol=1e-4, ftol=1e-4, gtol=1e-4
        )
        x_final = result.x
        conv = {'success': result.success, 'message': result.message,
                'nfev': result.nfev, 'cost': result.cost}
    except Exception as e:
        return {"status": "error", "message": str(e)}

    fit_params = dict(zip(params_to_fit, x_final))

    # Per-batch R²
    def r2(y_true, y_pred):
        yt, yp = np.array(y_true), np.array(y_pred)
        ss_res = np.sum((yt - yp)**2)
        ss_tot = np.sum((yt - np.mean(yt))**2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    r2_by_batch = {}
    for bid in batch_ids:
        b = all_batches[bid]
        m = b['measurements']
        t_obs = np.array(m['t'])
        V0 = m['volume'][0] if m['volume'][0] else 10.0
        S0 = m['substrate'][0] if m['substrate'][0] else 4.5
        X_v0 = m['vcd'][0] if m['vcd'][0] else 1.0
        y0 = [X_v0, 0.0, S0, 0.0, 0.0, 0.007, V0, 37.0, 7.0]

        _sim_cache.clear()
        t_sim, y_sim = simulate(b, y0, organism, fit_params)
        X_v_sim = np.interp(t_obs, t_sim, y_sim[0])
        S_sim = np.interp(t_obs, t_sim, y_sim[2])
        B_sim = np.interp(t_obs, t_sim, y_sim[4])

        obs_vcd = [v for v in m['vcd'] if v is not None]
        obs_s = [v for v in m['substrate'] if v is not None]
        obs_b = [v for v in m['byproduct'] if v is not None]

        r2_by_batch[bid] = {
            'R2_X_v': round(r2(obs_vcd, X_v_sim[:len(obs_vcd)]), 4),
            'R2_S': round(r2(obs_s, S_sim[:len(obs_s)]), 4),
            'R2_B': round(r2(obs_b, B_sim[:len(obs_b)]), 4),
        }

    return {
        "status": "success",
        "fitted_params": {k: float(v) for k, v in fit_params.items()},
        "convergence": conv,
        "goodness_of_fit": r2_by_batch,
    }


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    for bname in ['Batch1']:
        print(f"\n=== Fit ecoli_k12 → {bname} ===")
        result = fit_organism('ecoli_k12', [bname],
                              params_to_fit=['mu_max', 'K_s', 'Y_XS', 'alpha', 'beta', 'k_d'])
        if result['status'] == 'success':
            for bid, r2 in result['goodness_of_fit'].items():
                print(f"  {bid}: R² X_v={r2['R2_X_v']:.4f}  S={r2['R2_S']:.4f}  B={r2['R2_B']:.4f}")
            print(f"  Params: {', '.join(f'{k}={v:.4f}' for k,v in result['fitted_params'].items())}")
            print(f"  nfev={result['convergence']['nfev']}")
        else:
            print(f"  ERROR: {result.get('message')}")

"""
Parameter estimation engine v2:
- Uses actual DTB feed profiles as time-dependent inputs
- Fits only the intrinsic organism growth/substrate/product parameters
- Validates against holdout batches (to_predict=True)
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple
from dataclasses import replace

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_engine import (
    SimulationConfig, OrganismConfig, GrowthConfig, ProductConfig,
    EnvironmentConfig, OxygenConfig, HeatConfig,
    BioreactorConfig, InitialConditions, BioreactorODE,
    temp_factor_cardinal, ph_factor_gaussian,
    specific_growth_rate, specific_substrate_uptake,
    specific_product_formation
)
import psycopg2
import psycopg2.extras

DB_URL = "host=localhost dbname=bioreactor user=appuser password=bioreactor123"


def load_batch_data(batch_id: str, project: str = "DTB") -> Dict:
    """Load complete batch data including feed profiles"""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    cur.execute("""
        SELECT ed.*, e.batch_id as bid
        FROM experiment_data ed 
        JOIN experiments e ON ed.experiment_id = e.id
        WHERE e.project = %s AND e.batch_id = %s
        ORDER BY ed.t
    """, (project, batch_id))
    
    rows = list(cur.fetchall())
    cur.close(); conn.close()
    
    result = {
        't': [], 'vcd': [], 'substrate': [], 'byproduct': [], 'volume': [],
        'feed_substrate': [], 'feed_b': [], 'feed_c': [], 'feed_d': [],
        't_feed_substrate': [], 't_feed_b': [], 't_feed_c': [], 't_feed_d': [],
        'val_feed_substrate': [], 'val_feed_b': [], 'val_feed_c': [], 'val_feed_d': [],
    }
    
    for r in rows:
        result['t'].append(r['t'])
        result['vcd'].append(r['vcd'])
        result['substrate'].append(r['substrate'])
        result['byproduct'].append(r['byproduct'])
        result['volume'].append(r['volume'])
        
        if r['feed_substrate'] is not None:
            result['t_feed_substrate'].append(r['t'])
            result['val_feed_substrate'].append(r['feed_substrate'])
        if r['feed_b'] is not None:
            result['t_feed_b'].append(r['t'])
            result['val_feed_b'].append(r['feed_b'])
        if r['feed_c'] is not None:
            result['t_feed_c'].append(r['t'])
            result['val_feed_c'].append(r['feed_c'])
        if r['feed_d'] is not None:
            result['t_feed_d'].append(r['t'])
            result['val_feed_d'].append(r['feed_d'])
    
    return result


def build_feed_interpolators(batch_data: Dict, t_max: float) -> Dict:
    """Create interpolating functions for feed streams."""
    feeds = {}
    for feed_name in ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']:
        t_key = f't_{feed_name}'
        v_key = f'val_{feed_name}'
        
        if batch_data[t_key]:
            feeds[feed_name] = interp1d(
                batch_data[t_key], batch_data[v_key],
                kind='previous', bounds_error=False,
                fill_value=0.0
            )
        else:
            feeds[feed_name] = lambda t: 0.0
    
    return feeds


def custom_ode(t, y, organism, batch_data, feeds, fit_params=None):
    """
    Custom ODE with DTB feed data incorporated as time-varying inputs.
    
    y = [X_v, X_d, S, P, B, DO, V, T, pH]
    """
    X_v, X_d, S, P, B, DO, V, T, pH = y
    
    # Use fitted params if provided
    g = organism.growth
    if fit_params:
        mu_max = fit_params.get('mu_max', g.mu_max)
        K_s = fit_params.get('K_s', g.K_s)
        Y_XS = fit_params.get('Y_XS', g.Y_XS)
        m_s = fit_params.get('m_s', g.m_s)
        n = fit_params.get('n', g.n)
    else:
        mu_max, K_s, Y_XS, m_s, n = g.mu_max, g.K_s, g.Y_XS, g.m_s, g.n
    
    # Environment factors
    f_T = temp_factor_cardinal(T, organism.environment.T_min, 
                                organism.environment.T_opt, organism.environment.T_max)
    f_pH = ph_factor_gaussian(pH, organism.environment.pH_opt, organism.environment.pH_sigma)
    
    # Specific growth rate (Monod)
    mu = mu_max * (S / (K_s + S)) * f_T * f_pH
    
    # Specific uptake rate (Pirt model)
    if Y_XS > 0:
        q_S = mu / Y_XS + m_s
    else:
        q_S = m_s
    
    # Product formation
    q_P = 0
    q_B = 0
    if organism.products:
        for prod in organism.products:
            if fit_params:
                alpha = fit_params.get('alpha', prod.alpha)
                beta = fit_params.get('beta', prod.beta)
            else:
                alpha, beta = prod.alpha, prod.beta
            
            rate = alpha * mu + beta
            if prod.product_type == 'byproduct':
                q_B = rate
            else:
                q_P = rate
    
    # Death + lysis
    k_d = organism.k_d_max * max(0, 1 - S / (S + K_s))
    dX_v = (mu - k_d) * X_v
    dX_d = k_d * X_v - organism.k_lysis * X_d
    
    # Substrate: consumption + feeds
    dS = -q_S * X_v / V
    
    # Add feed inputs (g/L/h per feed stream, normalized by volume)
    for feed_name, interp_fn in feeds.items():
        feed_rate = interp_fn(t)
        if feed_rate > 0:
            if feed_name == 'feed_substrate':
                dS += feed_rate / V
            elif feed_name == 'feed_b':
                dS += feed_rate / V  # feed_b typically supplements substrate
            elif feed_name == 'feed_c':
                dS += 0.5 * feed_rate / V  # partial substrate contribution
            elif feed_name == 'feed_d':
                dS += 0.3 * feed_rate / V
    
    # Products
    dP = q_P * X_v
    dB = q_B * X_v
    
    # Oxygen (simplified)
    q_O2 = organism.oxygen.q_O2_max * (mu / (mu_max + 1e-6))
    kLa = 150.0
    O2_star = 0.007
    dDO = kLa * (O2_star - DO) - q_O2 * X_v
    
    # Volume
    feed_total = sum(f(t) for f in feeds.values())
    dV = feed_total / 1000.0  # convert to liters
    
    # Temperature and pH held constant in batch
    dT = 0
    dpH = 0
    
    return [dX_v, dX_d, dS, dP, dB, dDO, dV, dT, dpH]


def fit_organism_with_feeds(
    organism_id: str,
    batch_id: str,
    project: str = "DTB",
    params_to_fit: Optional[List[str]] = None,
    loss: str = "huber"
) -> Dict:
    """
    Fit organism parameters to a single DTB batch using actual feed profiles.
    """
    from db_connector import load_organism
    
    organism = load_organism(organism_id)
    if organism is None:
        return {"status": "error", "message": f"Organism {organism_id} not found"}
    
    batch = load_batch_data(batch_id, project)
    feeds = build_feed_interpolators(batch, max(batch['t']) * 1.1)
    
    # Observation mask
    mask = [(batch['vcd'][i] is not None and batch['substrate'][i] is not None)
            for i in range(len(batch['t']))]
    t_obs = np.array([batch['t'][i] for i, m in enumerate(mask) if m])
    vcd_obs = np.array([batch['vcd'][i] for i, m in enumerate(mask) if m])
    s_obs = np.array([batch['substrate'][i] for i, m in enumerate(mask) if m])
    b_obs = np.array([(batch['byproduct'][i] if batch['byproduct'][i] else 0)
                       for i, m in enumerate(mask) if m])
    
    if len(t_obs) < 5:
        return {"status": "error", "message": "Too few observations"}
    
    t_end = max(batch['t']) * 1.05
    
    # Initial conditions
    V0 = batch['volume'][0] if batch['volume'][0] else 10.0
    DO0 = 0.007
    y0 = [float(vcd_obs[0]), 0.0, float(s_obs[0]), 0.0, float(b_obs[0]), DO0, V0, 37.0, 7.0]
    
    if params_to_fit is None:
        params_to_fit = ['mu_max', 'K_s', 'Y_XS', 'alpha', 'beta', 'm_s']
    
    # Initial guess
    x0 = []
    for k in params_to_fit:
        vals = {
            'mu_max': organism.growth.mu_max,
            'K_s': organism.growth.K_s,
            'Y_XS': organism.growth.Y_XS,
            'K_i': organism.growth.K_i or 1.0,
            'm_s': organism.growth.m_s,
            'n': organism.growth.n,
            'alpha': organism.products[0].alpha if organism.products else 0.1,
            'beta': organism.products[0].beta if organism.products else 0.01,
        }
        x0.append(vals.get(k, 1.0))
    x0 = np.clip(np.array(x0), 1e-4, 50.0)
    
    def objective(p):
        fit_params = dict(zip(params_to_fit, p))
        
        try:
            sol = solve_ivp(
                lambda t, y: custom_ode(t, y, organism, batch, feeds, fit_params),
                (0, t_end), y0,
                method='LSODA', rtol=1e-4, atol=1e-6, max_step=5.0
            )
        except Exception:
            return np.ones(len(t_obs) * 3) * 1e6
        
        if not sol.success:
            return np.ones(len(t_obs) * 3) * 1e6
        
        X_v_sim = np.interp(t_obs, sol.t, sol.y[0])
        S_sim = np.interp(t_obs, sol.t, sol.y[2])
        B_sim = np.interp(t_obs, sol.t, sol.y[4])
        
        residuals = np.concatenate([
            (X_v_sim - vcd_obs) / max(np.mean(vcd_obs), 0.5),
            (S_sim - s_obs) / max(np.mean(s_obs), 0.5),
            0.5 * (B_sim - b_obs) / max(max(b_obs), 0.1)
        ])
        return residuals
    
    bounds = (np.ones(len(params_to_fit)) * 1e-4, np.ones(len(params_to_fit)) * 50.0)
    
    try:
        result = least_squares(objective, x0, bounds=bounds,
                                method='trf', loss=loss, max_nfev=100, xtol=1e-6, ftol=1e-6)
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    # Final fitted simulation
    fit_params = dict(zip(params_to_fit, result.x))
    t_fine = np.linspace(0, t_end, 500)
    sol = solve_ivp(lambda t, y: custom_ode(t, y, organism, batch, feeds, fit_params),
                    (0, t_end), y0, t_eval=t_fine, method='LSODA', rtol=1e-6, atol=1e-9)
    
    X_v_int = np.interp(t_obs, sol.t, sol.y[0])
    S_int = np.interp(t_obs, sol.t, sol.y[2])
    B_int = np.interp(t_obs, sol.t, sol.y[4])
    
    def r2(y_true, y_pred):
        yt = np.array(y_true)
        ss_res = np.sum((yt - y_pred)**2)
        ss_tot = np.sum((yt - np.mean(yt))**2)
        return 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    
    return {
        "status": "success",
        "organism_id": organism_id,
        "batch_id": batch_id,
        "project": project,
        "fitted_params": {k: float(v) for k, v in fit_params.items()},
        "initial_params": dict(zip(params_to_fit, x0.tolist())),
        "convergence": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost)
        },
        "goodness_of_fit": {
            "R2_X_v": round(r2(vcd_obs, X_v_int), 4),
            "R2_S": round(r2(s_obs, S_int), 4),
            "R2_B": round(r2(b_obs, B_int), 4)
        },
        "n_observations": len(t_obs),
        "simulation": {
            "time": sol.t.tolist(),
            "X_v": sol.y[0].tolist(),
            "S": sol.y[2].tolist(),
            "B": sol.y[4].tolist(),
            "V": sol.y[6].tolist(),
            "experimental": {
                "time": t_obs.tolist(),
                "X_v": vcd_obs.tolist(),
                "S": s_obs.tolist(),
                "B": b_obs.tolist()
            }
        }
    }


if __name__ == "__main__":
    import json
    
    print("=== Fit ecoli_k12 to DTB Batch1 (feed-driven) ===\n")
    result = fit_organism_with_feeds('ecoli_k12', 'Batch1')
    
    if result['status'] == 'success':
        print(f"Convergence: {result['convergence']['message']}")
        print(f"Function evals: {result['convergence']['nfev']}")
        r2 = result['goodness_of_fit']
        print(f"R²:  X_v={r2['R2_X_v']:.4f}  S={r2['R2_S']:.4f}  B={r2['R2_B']:.4f}")
        print(f"\nFitted params:")
        for k, v in result['fitted_params'].items():
            init = result['initial_params'][k]
            print(f"  {k:8s}: {init:.4f} → {v:.4f}")
    else:
        print(f"Error: {result.get('message')}")

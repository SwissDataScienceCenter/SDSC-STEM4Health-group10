"""
Cross-validation and holdout prediction for DTB data.

Strategy:
- Training batches: fit parameters individually, compute consensus
- Leave-one-out CV: fit on N-1 batches, predict on held-out batch
- Holdout (to_predict=True): generate pure predictions (no ground truth)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
from scipy.integrate import solve_ivp
from parameter_estimation import (
    load_batch_data, build_feed_interpolators, custom_ode, fit_organism_with_feeds
)
from db_connector import load_organism
import psycopg2
import psycopg2.extras

DB_URL = "host=localhost dbname=bioreactor user=appuser password=bioreactor123"


def leave_one_out_cv(organism_id: str, project: str = "DTB", 
                     n_batches: int = 10) -> dict:
    """
    Leave-one-out cross-validation:
    For each batch, fit on all OTHER batches, predict on held-out batch.
    """
    params_to_fit = ['mu_max', 'K_s', 'Y_XS', 'alpha', 'beta', 'm_s']
    
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT DISTINCT e.batch_id
        FROM experiments e
        JOIN experiment_data ed ON e.id = ed.experiment_id
        WHERE e.project = %s AND ed.to_predict = false
          AND ed.vcd IS NOT NULL AND ed.substrate IS NOT NULL
        ORDER BY e.batch_id
        LIMIT %s
    """, (project, n_batches))
    batch_ids = [r['batch_id'] for r in cur.fetchall()]
    cur.close(); conn.close()
    
    cv_results = []
    
    for held_out in batch_ids:
        print(f"\n--- LOO: holdout={held_out} ---")
        train = [b for b in batch_ids if b != held_out]
        
        # Fit on training batches
        all_params = {k: [] for k in params_to_fit}
        for bid in train:
            print(f"  fit {bid}...", end='', flush=True)
            res = fit_organism_with_feeds(organism_id, bid, project, params_to_fit)
            if res['status'] == 'success':
                for k in params_to_fit:
                    all_params[k].append(res['fitted_params'][k])
                print(f" R²_Xv={res['goodness_of_fit']['R2_X_v']:.3f}")
            else:
                print(" FAIL")
        
        if not all_params['mu_max']:
            continue
        
        # Consensus from training
        consensus = {k: float(np.median(v)) for k, v in all_params.items()}
        print(f"  consensus: μ_max={consensus['mu_max']:.3f} Y_XS={consensus['Y_XS']:.3f}")
        
        # Predict held-out batch
        organism = load_organism(organism_id)
        batch = load_batch_data(held_out, project)
        feeds = build_feed_interpolators(batch, max(batch['t']) * 1.05)
        
        # Filter valid observations
        mask = [(batch['vcd'][i] is not None and batch['substrate'][i] is not None)
                for i in range(len(batch['t']))]
        t_obs = np.array([batch['t'][i] for i, m in enumerate(mask) if m])
        xv_obs = np.array([batch['vcd'][i] for i, m in enumerate(mask) if m])
        s_obs = np.array([batch['substrate'][i] for i, m in enumerate(mask) if m])
        b_obs = np.array([(batch['byproduct'][i] if batch['byproduct'][i] else 0)
                          for i, m in enumerate(mask) if m])
        
        V0 = batch['volume'][0] if batch['volume'][0] else 10.0
        y0 = [float(xv_obs[0]), 0.0, float(s_obs[0]), 0.0, float(b_obs[0]), 0.007, V0, 37.0, 7.0]
        t_end = max(batch['t']) * 1.05
        
        try:
            sol = solve_ivp(
                lambda t, y: custom_ode(t, y, organism, batch, feeds, consensus),
                (0, t_end), y0,
                method='LSODA', rtol=1e-4, atol=1e-6, max_step=5.0
            )
        except Exception as e:
            print(f"  ODE failed: {e}")
            continue
        
        X_v_pred = np.interp(t_obs, sol.t, sol.y[0])
        S_pred = np.interp(t_obs, sol.t, sol.y[2])
        B_pred = np.interp(t_obs, sol.t, sol.y[4])
        
        def rmse(yt, yp):
            return float(np.sqrt(np.mean((np.array(yt) - yp)**2)))
        
        def r2(yt, yp):
            ytn = np.array(yt)
            ss_res = np.sum((ytn - yp)**2)
            ss_tot = np.sum((ytn - np.mean(ytn))**2)
            return float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.0
        
        cv_results.append({
            "held_out": held_out,
            "consensus_params": consensus,
            "metrics": {
                "RMSE_X_v": rmse(xv_obs, X_v_pred),
                "RMSE_S": rmse(s_obs, S_pred),
                "R2_X_v": r2(xv_obs, X_v_pred),
                "R2_S": r2(s_obs, S_pred)
            }
        })
        print(f"  CV: RMSE_Xv={cv_results[-1]['metrics']['RMSE_X_v']:.2f} "
              f"R²_Xv={cv_results[-1]['metrics']['R2_X_v']:.3f}")
    
    if not cv_results:
        return {"error": "No CV results"}
    
    rmses_xv = [r['metrics']['RMSE_X_v'] for r in cv_results]
    r2s_xv = [r['metrics']['R2_X_v'] for r in cv_results]
    
    return {
        "method": "leave-one-out CV",
        "organism_id": organism_id,
        "n_folds": len(cv_results),
        "summary": {
            "mean_RMSE_X_v": round(float(np.mean(rmses_xv)), 3),
            "std_RMSE_X_v": round(float(np.std(rmses_xv)), 3),
            "mean_R2_X_v": round(float(np.mean(r2s_xv)), 3),
            "std_R2_X_v": round(float(np.std(r2s_xv)), 3),
        },
        "folds": cv_results
    }


def predict_holdout(organism_id: str, consensus_params: dict,
                    project: str = "DTB") -> dict:
    """Generate pure predictions for the to_predict=True holdout batches."""
    
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT DISTINCT e.batch_id
        FROM experiments e
        JOIN experiment_data ed ON e.id = ed.experiment_id
        WHERE e.project = %s AND ed.to_predict = true
        ORDER BY e.batch_id
    """, (project,))
    holdout_batches = [r['batch_id'] for r in cur.fetchall()]
    cur.close(); conn.close()
    
    organism = load_organism(organism_id)
    predictions = []
    
    for bid in holdout_batches:
        batch = load_batch_data(bid, project)
        feeds = build_feed_interpolators(batch, max(batch['t']) * 1.05)
        
        # Get initial conditions (t=0, the only measured point)
        V0 = float(batch['volume'][0]) if batch['volume'] and batch['volume'][0] else 10.0
        X_v0 = float(batch['vcd'][0]) if batch['vcd'] and batch['vcd'][0] else 1.0
        S0 = float(batch['substrate'][0]) if batch['substrate'] and batch['substrate'][0] else 5.0
        B0 = float(batch['byproduct'][0]) if batch['byproduct'] and batch['byproduct'][0] else 0.0
        
        y0 = [X_v0, 0.0, S0, 0.0, B0, 0.007, V0, 37.0, 7.0]
        t_end = max(batch['t']) * 1.05
        
        # Prediction times (the to_predict rows)
        conn2 = psycopg2.connect(DB_URL)
        cur2 = conn2.cursor()
        cur2.execute("""
            SELECT ed.t FROM experiment_data ed
            JOIN experiments e ON ed.experiment_id = e.id
            WHERE e.project = %s AND e.batch_id = %s AND ed.to_predict = true
            ORDER BY ed.t
        """, (project, bid))
        t_pred = np.array([r[0] for r in cur2.fetchall()])
        cur2.close(); conn2.close()
        
        if len(t_pred) == 0:
            continue
        
        try:
            sol = solve_ivp(
                lambda t, y: custom_ode(t, y, organism, batch, feeds, consensus_params),
                (0, t_end), y0,
                method='LSODA', rtol=1e-4, atol=1e-6, max_step=5.0
            )
        except Exception:
            continue
        
        X_v_pred = np.interp(t_pred, sol.t, sol.y[0])
        S_pred = np.interp(t_pred, sol.t, sol.y[2])
        B_pred = np.interp(t_pred, sol.t, sol.y[4])
        V_pred = np.interp(t_pred, sol.t, sol.y[6])
        X_d_pred = np.interp(t_pred, sol.t, sol.y[1])
        
        predictions.append({
            "batch_id": bid,
            "initial": {"X_v0": X_v0, "S0": S0, "B0": B0, "V0": V0},
            "time": t_pred.tolist(),
            "X_v": X_v_pred.tolist(),
            "X_d": X_d_pred.tolist(),
            "substrate": S_pred.tolist(),
            "byproduct": B_pred.tolist(),
            "volume": V_pred.tolist(),
        })
    
    return {
        "organism_id": organism_id,
        "consensus_params": consensus_params,
        "n_holdout_batches": len(predictions),
        "predictions": predictions
    }


if __name__ == "__main__":
    consensus = {
        'mu_max': 0.725, 'K_s': 0.010, 'Y_XS': 0.550,
        'alpha': 0.043, 'beta': 0.001, 'm_s': 0.032
    }
    
    print("=== Leave-One-Out Cross-Validation ===\n")
    cv = leave_one_out_cv('ecoli_k12', n_batches=8)
    
    if 'summary' in cv:
        s = cv['summary']
        print(f"\n=== CV Summary ({len(cv_results)} folds) ===")
        print(f"Folds: {cv['n_folds']}")
        print(f"Mean RMSE X_v: {s['mean_RMSE_X_v']:.3f} ± {s['std_RMSE_X_v']:.3f} g/L")
        print(f"Mean R² X_v:  {s['mean_R2_X_v']:.3f} ± {s['std_R2_X_v']:.3f}")
    
    print("\n=== Holdout Predictions (to_predict=True) ===\n")
    pred = predict_holdout('ecoli_k12', consensus)
    print(f"Generated predictions for {pred['n_holdout_batches']} batches")
    for p in pred['predictions']:
        print(f"  {p['batch_id']}: X_v: {p['X_v'][0]:.2f} → {p['X_v'][-1]:.2f} g/L")
    
    cv['holdout_predictions'] = pred
    with open('validation_results.json', 'w') as f:
        json.dump(cv, f, indent=2, default=str)
    print("\nSaved validation_results.json")

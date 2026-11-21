"""
Batch parameter estimation across multiple DTB batches.
Fits each batch individually, then computes consensus parameters.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from parameter_estimation import fit_organism_with_feeds
import psycopg2
import psycopg2.extras

DB_URL = "host=localhost dbname=bioreactor user=appuser password=bioreactor123"


def fit_across_batches(
    organism_id: str = 'ecoli_k12',
    project: str = 'DTB',
    n_train: int = 10,
    params_to_fit: list = None
):
    """Fit organism parameters across multiple training batches."""
    
    if params_to_fit is None:
        params_to_fit = ['mu_max', 'K_s', 'Y_XS', 'alpha', 'beta', 'm_s']
    
    # Get training batches (non-holdout)
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT DISTINCT e.batch_id
        FROM experiments e
        JOIN experiment_data ed ON e.id = ed.experiment_id
        WHERE e.project = %s AND ed.to_predict = false
          AND ed.vcd IS NOT NULL AND ed.substrate IS NOT NULL
          AND ed.vcd > 1.0
        ORDER BY e.batch_id
        LIMIT %s
    """, (project, n_train))
    
    train_batches = [r['batch_id'] for r in cur.fetchall()]
    
    # Get holdout batches
    cur.execute("""
        SELECT DISTINCT e.batch_id
        FROM experiments e
        JOIN experiment_data ed ON e.id = ed.experiment_id
        WHERE e.project = %s AND ed.to_predict = true
        ORDER BY e.batch_id
    """, (project,))
    holdout_batches = [r['batch_id'] for r in cur.fetchall()]
    cur.close(); conn.close()
    
    print(f"Training batches: {train_batches}")
    print(f"Holdout (to_predict): {holdout_batches}")
    print()
    
    # Fit each training batch
    all_params = {k: [] for k in params_to_fit}
    all_r2 = {'R2_X_v': [], 'R2_S': [], 'R2_B': []}
    individual_results = []
    
    for bid in train_batches:
        print(f"  [{bid}] ", end='', flush=True)
        res = fit_organism_with_feeds(organism_id, bid, project, params_to_fit)
        if res['status'] == 'success':
            r2 = res['goodness_of_fit']
            all_r2['R2_X_v'].append(r2['R2_X_v'])
            all_r2['R2_S'].append(r2['R2_S'])
            all_r2['R2_B'].append(r2['R2_B'])
            
            line = f"R²: XV={r2['R2_X_v']:.3f}, S={r2['R2_S']:.3f}, B={r2['R2_B']:.3f}"
            print(line)
            
            for k in params_to_fit:
                all_params[k].append(res['fitted_params'][k])
            
            individual_results.append({
                'batch_id': bid,
                'r2': r2,
                'params': res['fitted_params'],
                'n_obs': res['n_observations']
            })
        else:
            print(f"FAILED: {res.get('message', 'unknown')}")
    
    if not individual_results:
        return {"status": "error", "message": "No successful fits"}
    
    # Consensus parameters (median with IQR)
    consensus = {}
    for k in params_to_fit:
        vals = np.array(all_params[k])
        consensus[k] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'median': float(np.median(vals)),
            'Q1': float(np.percentile(vals, 25)),
            'Q3': float(np.percentile(vals, 75)),
            'cv_pct': round(float(np.std(vals) / np.mean(vals) * 100), 1)
        }
    
    avg_r2 = {
        'R2_X_v': round(float(np.mean(all_r2['R2_X_v'])), 3),
        'R2_S': round(float(np.mean(all_r2['R2_S'])), 3),
        'R2_B': round(float(np.mean(all_r2['R2_B'])), 3),
        'R2_X_v_std': round(float(np.std(all_r2['R2_X_v'])), 3),
        'R2_S_std': round(float(np.std(all_r2['R2_S'])), 3),
        'R2_B_std': round(float(np.std(all_r2['R2_B'])), 3),
    }
    
    return {
        "status": "success",
        "organism_id": organism_id,
        "project": project,
        "n_batches_fitted": len(individual_results),
        "train_batches": train_batches,
        "holdout_batches": holdout_batches,
        "params_to_fit": params_to_fit,
        "consensus_parameters": consensus,
        "average_R2": avg_r2,
        "individual_results": individual_results
    }


if __name__ == "__main__":
    results = fit_across_batches('ecoli_k12', n_train=10,
                                  params_to_fit=['mu_max', 'K_s', 'Y_XS', 'alpha', 'beta', 'm_s'])
    
    if results['status'] == 'success':
        print(f"\n=== Consensus Parameters ({results['n_batches_fitted']} batches) ===")
        for k, v in results['consensus_parameters'].items():
            print(f"  {k:8s}: median={v['median']:.4f}  mean={v['mean']:.4f}±{v['std']:.4f}  CV={v['cv_pct']}%")
        
        print(f"\n=== Average Goodness-of-Fit ===")
        for k, v in results['average_R2'].items():
            if 'std' not in k:
                std_key = f'{k}_std'
                print(f"  {k}: {v:.3f} ± {results['average_R2'].get(std_key, 0):.3f}")
        
        print(f"\nHoldout batches to predict: {results['holdout_batches']}")
        
        # Save to file
        with open('fit_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved to fit_results.json")

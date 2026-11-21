#!/usr/bin/env python3
"""
Save fitted parameters back to PostgreSQL and update organism configurations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psycopg2
import psycopg2.extras
import json
from datetime import datetime

DB_URL = "host=localhost dbname=bioreactor user=appuser password=bioreactor123"

# Consensus parameters from 9-batch fitting
ECOLI_K12_CONSENSUS = {
    'mu_max': 0.725,   # h⁻¹
    'K_s': 0.010,      # g/L
    'Y_XS': 0.550,     # g/g
    'alpha': 0.043,    # g/g (growth-associated acetate)
    'beta': 0.001,     # g/g/h (non-growth-associated acetate)
    'm_s': 0.032,      # g/g/h (maintenance coefficient)
}


def update_organism_params(organism_id: str, params: dict, source: str = "DTB fitting"):
    """Update organism parameters in PostgreSQL with fitted values."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # Map fitted params to DB columns
    column_map = {
        'mu_max': 'mu_max',
        'K_s': 'k_s',
        'K_i': 'k_i',
        'Y_XS': 'y_xs',
        'm_s': 'm_s',
        'n': 'n',
    }
    
    # Update growth params
    growth_cols = {k: v for k, v in params.items() if k in column_map}
    if growth_cols:
        set_clause = ', '.join(f"{column_map[k]} = %({k})s" for k in growth_cols)
        cur.execute(f"""
            INSERT INTO organism_growth_params (organism_id, model, {', '.join(column_map[k] for k in growth_cols)})
            VALUES (%(org_id)s, %(model)s, {', '.join(f'%({k})s' for k in growth_cols)})
            ON CONFLICT (organism_id) 
            DO UPDATE SET {set_clause}
        """, {**growth_cols, 'org_id': organism_id, 'model': 'monod'})
    
    # Update product (acetate) params
    if 'alpha' in params or 'beta' in params:
        cur.execute("SELECT id FROM organism_products WHERE organism_id = %s AND product_name = 'acetate'",
                    (organism_id,))
        row = cur.fetchone()
        if row:
            updates = []
            if 'alpha' in params:
                updates.append(f"alpha = {params['alpha']}")
            if 'beta' in params:
                updates.append(f"beta = {params['beta']}")
            if updates:
                cur.execute(f"UPDATE organism_products SET {', '.join(updates)} WHERE id = %s", (row[0],))
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"  Updated {organism_id}: {json.dumps(params, indent=2)}")


if __name__ == "__main__":
    print("=== Saving Fitted Parameters to DB ===\n")
    
    # Update E. coli K-12 with consensus params
    update_organism_params('ecoli_k12', ECOLI_K12_CONSENSUS, 'DTB 9-batch consensus')
    
    # Verify
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT ogp.mu_max, ogp.k_s, ogp.y_xs, ogp.m_s, op.alpha, op.beta
        FROM organism_growth_params ogp
        LEFT JOIN organism_products op ON ogp.organism_id = op.organism_id AND op.product_name = 'acetate'
        WHERE ogp.organism_id = 'ecoli_k12'
    """)
    row = cur.fetchone()
    cur.close(); conn.close()
    
    print("\n=== Verification ===")
    print(f"  mu_max: {row['mu_max']}")
    print(f"  K_s:    {row['k_s']}")
    print(f"  Y_XS:   {row['y_xs']}")
    print(f"  m_s:    {row['m_s']}")
    print(f"  alpha:  {row['alpha']}")
    print(f"  beta:   {row['beta']}")
    print("\n✓ Parameters saved successfully")

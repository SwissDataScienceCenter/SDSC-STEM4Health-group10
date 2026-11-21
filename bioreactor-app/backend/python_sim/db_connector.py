"""
Database connector: loads organism configurations from PostgreSQL
and saves simulation results back.
"""

import psycopg2
import psycopg2.extras
import json
from typing import Optional, List, Dict
from simulation_engine import (
    OrganismConfig, GrowthConfig, SubstrateConfig, ProductConfig,
    EnvironmentConfig, OxygenConfig, HeatConfig, BioreactorConfig,
    InitialConditions, SimulationConfig, run_simulation
)


DB_URL = "host=localhost dbname=bioreactor user=appuser password=bioreactor123"


def get_connection():
    return psycopg2.connect(DB_URL)


def _val(d, key, default=None):
    """Get value from dict cursor row with fallback"""
    try:
        v = d[key]
        return v if v is not None else default
    except KeyError:
        return default


def load_organism(organism_id: str) -> Optional[OrganismConfig]:
    """Load full organism configuration from database"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    cur.execute("SELECT * FROM organisms WHERE id = %s", (organism_id,))
    org_row = cur.fetchone()
    if not org_row:
        cur.close(); conn.close()
        return None
    
    # Growth params
    cur.execute("SELECT * FROM organism_growth_params WHERE organism_id = %s", (organism_id,))
    g = cur.fetchone()
    growth = GrowthConfig(
        model=_val(g, 'model', 'monod'),
        mu_max=float(_val(g, 'mu_max', 0.7)),
        K_s=float(_val(g, 'k_s', 0.02)),
        K_i=float(_val(g, 'k_i')) if _val(g, 'k_i') else 50.0,
        Y_XS=float(_val(g, 'y_xs', 0.5)),
        m_s=float(_val(g, 'm_s', 0.025)),
        n=float(_val(g, 'n', 1.0))
    ) if g else GrowthConfig()
    
    # Substrates
    cur.execute("""
        SELECT os.*, s.name as s_name 
        FROM organism_substrates os 
        JOIN substrates s ON os.substrate_id = s.id 
        WHERE os.organism_id = %s ORDER BY os.priority
    """, (organism_id,))
    substrates = []
    for row in cur.fetchall():
        substrates.append(SubstrateConfig(
            name=row['s_name'],
            q_S_max=float(row['q_s_max']),
            K_m=float(row['k_m']),
            Y_XS=float(_val(row, 'y_xs', growth.Y_XS)),
            priority=row['priority']
        ))
    
    # Products
    cur.execute("SELECT * FROM organism_products WHERE organism_id = %s", (organism_id,))
    products = []
    for row in cur.fetchall():
        products.append(ProductConfig(
            name=row['product_name'],
            product_type=_val(row, 'product_type', 'byproduct'),
            model=_val(row, 'model', 'luedeking-piret'),
            alpha=float(_val(row, 'alpha', 0)),
            beta=float(_val(row, 'beta', 0)),
            Y_PS=float(_val(row, 'y_ps', 0)),
            q_S_crit=float(row['q_s_crit']) if _val(row, 'q_s_crit') else None,
            inhibition_type=_val(row, 'inhibition_type', 'none'),
            K_p=float(row['k_p']) if _val(row, 'k_p') else None
        ))
    
    # Environment
    cur.execute("SELECT * FROM organism_environment WHERE organism_id = %s", (organism_id,))
    e = cur.fetchone()
    env = EnvironmentConfig(
        temp_model=_val(e, 'temp_model', 'cardinal'),
        T_min=float(_val(e, 't_min', 8.0)),
        T_opt=float(_val(e, 't_opt', 37.0)),
        T_max=float(_val(e, 't_max', 45.0)),
        pH_min=float(_val(e, 'ph_min', 5.0)),
        pH_opt=float(_val(e, 'ph_opt', 7.0)),
        pH_max=float(_val(e, 'ph_max', 9.0)),
        pH_sigma=float(_val(e, 'ph_sigma', 0.8))
    ) if e else EnvironmentConfig()
    
    # Oxygen
    cur.execute("SELECT * FROM organism_oxygen_params WHERE organism_id = %s", (organism_id,))
    o = cur.fetchone()
    oxy = OxygenConfig(
        q_O2_max=float(_val(o, 'q_o2_max', 0.4)),
        Y_XO=float(_val(o, 'y_xo', 0.95)),
        m_O=float(_val(o, 'm_o', 0.01))
    ) if o else OxygenConfig()
    
    # Heat
    cur.execute("SELECT * FROM organism_heat_params WHERE organism_id = %s", (organism_id,))
    h = cur.fetchone()
    heat = HeatConfig(
        Y_QX=float(_val(h, 'y_qx', 19.0)),
        Y_QS=float(_val(h, 'y_qs', 14.0))
    ) if h else HeatConfig()
    
    # Death
    cur.execute("SELECT * FROM organism_death_params WHERE organism_id = %s", (organism_id,))
    d = cur.fetchone()
    k_d_max = float(d['k_d_max']) if d else 0.01
    k_lysis = float(d['k_lysis']) if d else 0.005
    
    cur.close(); conn.close()
    
    return OrganismConfig(
        name=org_row['name'],
        growth=growth,
        substrates=substrates,
        products=products,
        environment=env,
        oxygen=oxy,
        heat=heat,
        k_d_max=k_d_max,
        k_lysis=k_lysis
    )


def save_simulation_result(simulation_id: int, results: Dict, conn=None):
    """Save simulation time series to PostgreSQL"""
    should_close = conn is None
    if should_close:
        conn = get_connection()
    cur = conn.cursor()
    
    from io import StringIO
    buf = StringIO()
    for i in range(len(results['time'])):
        buf.write('\t'.join([
            str(simulation_id),
            str(results['time'][i]),
            f"{results['X_v'][i]:.6f}", f"{results['X_d'][i]:.6f}",
            f"{results['substrate'][i]:.6f}", f"{results['product'][i]:.6f}",
            f"{results['byproduct'][i]:.6f}", f"{results['dissolved_oxygen'][i]:.6f}",
            f"{results['volume'][i]:.6f}", f"{results['temperature'][i]:.6f}",
            f"{results['ph'][i]:.6f}",
            '\\N', '\\N', '\\N', '\\N', '\\N',
            f"{results['viability'][i]:.6f}",
        ]) + '\n')
    
    buf.seek(0)
    cur.copy_from(buf, 'simulation_results', columns=[
        'simulation_id','t','x_v','x_d','s','p','b','dissolved_o2',
        'v','temperature','ph','mu','q_s','q_p','q_b','q_o2','viability'
    ])
    conn.commit()
    if should_close:
        cur.close(); conn.close()


def create_simulation_from_db(organism_id: str, 
                               initial_conditions: Optional[Dict] = None,
                               t_end: float = 100.0) -> Optional[SimulationConfig]:
    """Create a complete simulation config from database records"""
    
    organism = load_organism(organism_id)
    if not organism:
        return None
    
    br = BioreactorConfig()
    ic = InitialConditions()
    if initial_conditions:
        ic = InitialConditions(**initial_conditions)
    
    return SimulationConfig(
        organism=organism,
        bioreactor=br,
        initial=ic,
        t_end=t_end,
        dt_output=0.1
    )


def list_organisms() -> List[Dict]:
    """List all available organisms in the database"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT id, name, organism_type, description FROM organisms ORDER BY name")
    organisms = [dict(row) for row in cur.fetchall()]
    cur.close(); conn.close()
    return organisms


def import_experiment_data(organism_id: str, project: str, batch_id: str) -> Optional[Dict]:
    """Import experimental data from DB for comparison with simulation"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    cur.execute("""
        SELECT ed.t, ed.vcd, ed.tcd, ed.substrate, ed.byproduct, ed.volume
        FROM experiment_data ed
        JOIN experiments e ON ed.experiment_id = e.id
        WHERE e.project = %s AND e.batch_id = %s
        ORDER BY ed.t
    """, (project, batch_id))
    
    rows = cur.fetchall()
    if not rows:
        cur.close(); conn.close()
        return None
    
    data = {
        'time': [r['t'] for r in rows],
        'vcd': [float(r['vcd']) if r['vcd'] else None for r in rows],
        'tcd': [float(r['tcd']) if r['tcd'] else None for r in rows],
        'substrate': [float(r['substrate']) if r['substrate'] else None for r in rows],
        'byproduct': [float(r['byproduct']) if r['byproduct'] else None for r in rows],
        'volume': [float(r['volume']) if r['volume'] else None for r in rows],
    }
    cur.close(); conn.close()
    return data


if __name__ == "__main__":
    print("=== Organisms in DB ===")
    for org in list_organisms():
        print(f"  {org['id']:15s} {org['name']}")
    
    print("\n=== E. coli K-12 from DB ===")
    config = create_simulation_from_db('ecoli_k12', 
        {'X_v': 1.0, 'S': 5.0, 'V': 10.0, 'temperature': 37.0, 'pH': 7.0},
        t_end=24.0)
    
    if config:
        print(f"Name: {config.organism.name}")
        print(f"μ_max: {config.organism.growth.mu_max} h⁻¹, K_s: {config.organism.growth.K_s} g/L")
        print(f"Substrates: {[(s.name, s.q_S_max) for s in config.organism.substrates]}")
        print(f"Products: {[(p.name, p.alpha, p.beta) for p in config.organism.products]}")
        
        print("\nRunning simulation...")
        results = run_simulation(config)
        print(f"  Status: {results['status']}")
        print(f"  Final X_v: {results['X_v'][-1]:.2f} g/L")
        print(f"  Final S: {results['substrate'][-1]:.3f} g/L")
        print(f"  Final B: {results['byproduct'][-1]:.3f} g/L")
        print(f"  Viability: {results['viability'][-1]:.3f}")

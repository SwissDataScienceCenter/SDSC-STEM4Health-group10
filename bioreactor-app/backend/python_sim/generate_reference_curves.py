"""
Generate reference curves from the Python simulation engine for Java port verification.
Creates simple scenarios that the Java MonodModel can replicate exactly.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))

from simulation_engine import (
    SimulationConfig, OrganismConfig, GrowthConfig, ProductConfig,
    SubstrateConfig, EnvironmentConfig, OxygenConfig, BioreactorConfig,
    InitialConditions, create_ecoli_batch_example, run_simulation
)

def run_and_save(config, name):
    result = run_simulation(config)
    out = os.path.join(os.path.dirname(__file__), f"ref_{name}.csv")
    with open(out, 'w') as f:
        w = csv.writer(f)
        w.writerow(['t','X_v','X_d','S','P','B','DO','V','T','pH'])
        for i in range(len(result['time'])):
            w.writerow([
                f"{result['time'][i]:.10f}",
                f"{result['X_v'][i]:.10f}",
                f"{result['X_d'][i]:.10f}",
                f"{result['substrate'][i]:.10f}",
                f"{result['product'][i]:.10f}",
                f"{result['byproduct'][i]:.10f}",
                f"{result['dissolved_oxygen'][i]:.10f}",
                f"{result['volume'][i]:.10f}",
                f"{result['temperature'][i]:.10f}",
                f"{result['pH'][i]:.10f}",
            ])
    print(f"Wrote {out}: {len(result['time'])} rows, status={result['status']}")

# 1. Pure batch (no product inhibition, no feeds, 24h)
batch_no_inhib = SimulationConfig(
    organism=OrganismConfig(
        name="Batch no inhibition",
        growth=GrowthConfig(
            model="monod", mu_max=0.70, K_s=0.02, Y_XS=0.50, m_s=0.025,
            K_i=0.0, K_p=0.0  # no inhibition
        ),
        substrates=[SubstrateConfig(name="glucose", q_S_max=1.5, K_m=0.01, Y_XS=0.50, priority=1)],
        products=[ProductConfig(name="lactate", product_type="byproduct", model="luedeking-piret",
                                 alpha=0.0, beta=0.02, Y_PS=0.15, q_S_crit=0.8,
                                 inhibition_type="linear", K_p=0.0)],  # K_p=0 = no inhibition
        environment=EnvironmentConfig(T_min=8.0, T_opt=37.0, T_max=45.0,
                                      pH_min=5.0, pH_opt=7.0, pH_max=9.0, pH_sigma=0.8),
        oxygen=OxygenConfig(q_O2_max=0.40, Y_XO=0.95, m_O=0.01),
        k_d_max=0.01, k_lysis=0.005
    ),
    bioreactor=BioreactorConfig(
        mode="batch", V_initial=10.0, kLa=150.0, O2_star=0.007,
        heat_transfer=0.5, T_setpoint=37.0
    ),
    initial=InitialConditions(X_v=1.0, S=5.0, V=10.0, temperature=37.0, pH=7.0),
    t_end=24.0,
    dt_output=0.1
)
run_and_save(batch_no_inhib, "batch_no_inhib")

# 2. E. coli batch (from library, has product inhibition)
ecoli = create_ecoli_batch_example()
run_and_save(ecoli, "ecoli_batch")

print("\nDone. Copy ref_*.csv to src/test/resources for Java comparison tests.")

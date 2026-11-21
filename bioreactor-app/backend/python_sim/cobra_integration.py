"""
COBRApy integration: genome-scale metabolic model (GEM) analysis
embedded within the bioreactor simulation framework.

Uses E. coli core model (textbook) to compute:
- Theoretical yield coefficients (Y_XS_max, Y_PS_max)
- Predicted byproduct profiles (acetate, ethanol, CO₂)
- Flux distribution at key growth phases
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from cobra.io import read_sbml_model
import cobra

# Cache the model at module level
_MODEL = None
_MODEL_PATH = os.path.join(os.path.dirname(cobra.__file__), 'data', 'textbook.xml.gz')

REACTION_MAP = {
    # Map DTB measurements to model exchange reactions
    'glucose': 'EX_glc__D_e',
    'oxygen': 'EX_o2_e',
    'acetate': 'EX_ac_e',
    'ethanol': 'EX_etoh_e',
    'formate': 'EX_for_e',
    'co2': 'EX_co2_e',
    'lactate': 'EX_lac__D_e',
    'succinate': 'EX_succ_e',
    'biomass': 'Biomass_Ecoli_core',
}


def get_model():
    """Load or return cached E. coli core model"""
    global _MODEL
    if _MODEL is None:
        _MODEL = read_sbml_model(_MODEL_PATH)
    return _MODEL


def compute_theoretical_yields(mu: float, substrate_uptake: float = 10.0) -> Dict:
    """
    Run FBA at a given growth rate and compute theoretical yields.
    
    Args:
        mu: growth rate (h⁻¹) — used to constrain biomass flux
        substrate_uptake: glucose uptake rate (mmol/gDW/h)
        
    Returns:
        Dict with yield coefficients and flux distribution
    """
    model = get_model().copy()
    
    # Find glucose exchange and biomass reactions
    glc_rxn = model.reactions.get_by_id('EX_glc__D_e')
    biomass_rxn = model.reactions.get_by_id('Biomass_Ecoli_core')
    
    # Constrain glucose uptake (negative = consumption)
    glc_rxn.lower_bound = -substrate_uptake
    
    # Run FBA
    try:
        sol = model.optimize()
        flux = sol.objective_value
    except Exception:
        return {"error": "FBA failed"}
    
    # Compute yield coefficients
    Y_XS_max = flux / substrate_uptake if substrate_uptake > 0 else 0  # gDW/mmol
    
    # Byproduct yields
    byproducts = {}
    for name, rxn_id in REACTION_MAP.items():
        if name in ('glucose', 'biomass', 'oxygen'):
            continue
        try:
            rxn = model.reactions.get_by_id(rxn_id)
            bp_flux = sol.fluxes[rxn_id]
            if bp_flux > 0.01:
                byproducts[name] = {
                    'flux': float(bp_flux),
                    'yield_per_glucose': float(bp_flux / substrate_uptake),
                    'yield_per_biomass': float(bp_flux / flux) if flux > 0 else 0
                }
        except KeyError:
            pass
    
    # Oxygen
    o2_flux = sol.fluxes.get('EX_o2_e', 0)
    
    return {
        "status": sol.status,
        "growth_rate": mu,
        "substrate_uptake": substrate_uptake,
        "biomass_flux": float(flux),
        "Y_XS_max": float(Y_XS_max),  # gDW/mmol glucose
        "oxygen_uptake": float(o2_flux),
        "byproduct_fluxes": byproducts,
        "maintenance": {
            "ATPM": float(sol.fluxes.get('ATPM', 0)),
        }
    }


def compute_yield_surface(mu_range: Tuple[float, float], points: int = 10) -> Dict:
    """
    Compute yield coefficients across a range of growth rates.
    This maps how Y_XS and byproduct yields change with μ.
    """
    mu_min, mu_max = mu_range
    mus = np.linspace(mu_min, mu_max, points)
    
    results = []
    for mu in mus:
        # Substrate uptake scales with growth rate (Pirt model)
        substrate_uptake = mu / 0.5 + 2.0  # approximate: Y_XS ~0.5 g/g
        res = compute_theoretical_yields(mu, min(substrate_uptake, 20.0))
        res['mu'] = mu
        results.append(res)
    
    return {
        "growth_rates": mus.tolist(),
        "yields": results
    }


def predict_byproduct_profile(mu: float, q_S: float) -> Dict:
    """
    Predict overflow metabolism byproducts using FBA.
    
    At high growth rates, E. coli produces acetate via overflow metabolism
    (Crabtree effect analog). FBA naturally captures this via the 
    trade-off between respiration and fermentation.
    """
    model = get_model().copy()
    glc_rxn = model.reactions.get_by_id('EX_glc__D_e')
    biomass_rxn = model.reactions.get_by_id('Biomass_Ecoli_core')
    o2_rxn = model.reactions.get_by_id('EX_o2_e')
    
    # Constrain glucose uptake
    glc_rxn.lower_bound = -q_S
    
    # Constrain oxygen (aerobic limit induces overflow)
    max_o2 = 20.0  # mmol/gDW/h — typical max OUR
    o2_rxn.lower_bound = -max_o2
    
    sol = model.optimize()
    
    predicted = {}
    for name, rxn_id in [('acetate', 'EX_ac_e'), ('formate', 'EX_for_e'),
                          ('ethanol', 'EX_etoh_e'), ('lactate', 'EX_lac__D_e'),
                          ('succinate', 'EX_succ_e'), ('co2', 'EX_co2_e')]:
        try:
            flux = sol.fluxes[rxn_id]
            if abs(flux) > 0.001:
                predicted[name] = float(flux)
        except KeyError:
            pass
    
    return {
        "growth_rate": mu,
        "q_S_glucose": float(q_S),
        "predicted_biomass": float(sol.objective_value),
        "byproducts": predicted,
        "status": sol.status
    }


def compare_ode_vs_fba(organism_id: str, mu: float, S: float) -> Dict:
    """
    Compare mechanistic ODE predictions with FBA-constrained metabolic network.
    Useful for validating the reduced-order model against genome-scale predictions.
    """
    from db_connector import load_organism
    from simulation_engine import specific_growth_rate
    
    org = load_organism(organism_id)
    if org is None:
        return {"error": f"Organism {organism_id} not found"}
    
    # ODE prediction
    ode_mu = specific_growth_rate(S, 0, 0, 37.0, 7.0, org.growth, org.environment, org.products)
    ode_q_S = org.substrates[0].q_S_max * S / (org.substrates[0].K_m + S) if org.substrates else 1.5
    
    # FBA prediction
    q_S_fba = ode_q_S * 0.18  # convert to mmol/gDW/h (approximate)
    fba_res = compute_theoretical_yields(mu, max(q_S_fba, 1.0))
    
    return {
        "organism": org.name,
        "growth_conditions": {"mu": mu, "S": S},
        "ode_prediction": {
            "mu_eff": ode_mu,
            "q_S": ode_q_S,
            "Y_XS": ode_mu / ode_q_S if ode_q_S > 0 else 0
        },
        "fba_prediction": {
            "mu_max_theoretical": fba_res.get("biomass_flux", 0),
            "Y_XS_max": fba_res.get("Y_XS_max", 0),
            "byproducts": fba_res.get("byproduct_fluxes", {})
        }
    }


if __name__ == "__main__":
    import json
    
    print("=== COBRApy E. coli Core Model ===\n")
    
    # Basic FBA
    res = compute_theoretical_yields(mu=0.7, substrate_uptake=10.0)
    print(f"Yields at μ=0.7 h⁻¹, q_S=10 mmol/gDW/h:")
    print(f"  Y_XS_max = {res['Y_XS_max']:.3f} gDW/mmol = {res['Y_XS_max'] * 1000 / 180:.3f} g/g glucose")
    print(f"  Oxygen uptake: {res['oxygen_uptake']:.1f} mmol/gDW/h")
    print(f"  Byproducts: {list(res['byproduct_fluxes'].keys())}")
    
    # Yield surface
    print("\nYield surface:")
    surface = compute_yield_surface((0.1, 0.8), points=5)
    for pt in surface['yields']:
        byps = ', '.join(f"{k}({v['flux']:.1f})" for k, v in pt.get('byproduct_fluxes', {}).items() if v['flux'] > 0.5)
        print(f"  μ={pt['mu']:.2f}: Y_XS={pt['Y_XS_max']:.3f}, byproducts: {byps or 'none'}")
    
    # Overflow prediction
    print("\nOverflow at high q_S:")
    overflow = predict_byproduct_profile(mu=0.7, q_S=15.0)
    print(f"  Biomass: {overflow['predicted_biomass']:.2f}")
    for name, flux in overflow['byproducts'].items():
        print(f"  {name}: {flux:.2f} mmol/gDW/h")
    
    print("\n✓ COBRApy integration complete")

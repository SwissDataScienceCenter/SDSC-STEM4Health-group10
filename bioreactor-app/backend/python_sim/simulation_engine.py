"""
Bioreactor Simulation Engine
ODE-based mechanistic model of microbial growth, substrate consumption,
product/byproduct formation, oxygen uptake, heat generation, and pH dynamics.

Architecture:
    Config (YAML/DB) → Model Builder → ODE System → scipy.integrate → Results
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json


# ======================================================================
# Configuration Data Classes
# ======================================================================

@dataclass
class GrowthConfig:
    model: str = "monod"          # monod, moser, andrews, haldane, aiba, logistic, double_monod
    mu_max: float = 0.7           # h⁻¹
    K_s: float = 0.02             # g/L
    K_i: float = 50.0             # g/L (inhibition)
    K_p: float = 10.0             # g/L (product inhibition)
    Y_XS: float = 0.50            # g biomass / g substrate
    m_s: float = 0.025            # g/g/h maintenance
    X_max: float = 100.0          # g/L (logistic)
    n: float = 1.0                # cooperativity (moser)

@dataclass
class SubstrateConfig:
    name: str = "glucose"
    q_S_max: float = 1.5          # g/g/h
    K_m: float = 0.01             # g/L
    Y_XS: float = 0.50            # g/g
    priority: int = 1

@dataclass
class ProductConfig:
    name: str = "acetate"
    product_type: str = "byproduct"  # product, byproduct
    model: str = "luedeking-piret"
    alpha: float = 0.0            # growth-associated
    beta: float = 0.02            # non-growth-associated (h⁻¹)
    Y_PS: float = 0.15            # g/g
    q_S_crit: float = 0.8         # g/g/h overflow threshold
    inhibition_type: str = "linear"
    K_p: float = 10.0             # g/L

@dataclass
class EnvironmentConfig:
    temp_model: str = "cardinal"
    T_min: float = 8.0
    T_opt: float = 37.0
    T_max: float = 45.0
    E_a: float = 50.0             # kJ/mol
    pH_min: float = 5.0
    pH_opt: float = 7.0
    pH_max: float = 9.0
    pH_sigma: float = 0.8

@dataclass
class OxygenConfig:
    q_O2_max: float = 0.4
    Y_XO: float = 0.95
    m_O: float = 0.01

@dataclass
class HeatConfig:
    Y_QX: float = 19.0            # kJ/g
    Y_QS: float = 14.0            # kJ/g

@dataclass
class BioreactorConfig:
    mode: str = "batch"           # batch, fed_batch, chemostat
    V_initial: float = 10.0       # L
    V_max: float = 20.0           # L
    kLa: float = 150.0            # h⁻¹
    O2_star: float = 0.007        # g/L
    T_setpoint: float = 37.0      # °C
    pH_setpoint: float = 7.0
    heat_transfer: float = 0.5    # kJ/h/K
    feed_profile: Optional[List[Tuple[float, float, float]]] = None  # [(time, flow_rate, S_feed), ...]

@dataclass
class InitialConditions:
    X_v: float = 1.0              # g/L viable biomass
    X_d: float = 0.0              # g/L dead biomass
    S: float = 5.0                # g/L substrate
    P: float = 0.0                # g/L product
    B: float = 0.0                # g/L byproduct
    dissolved_O2: float = 0.007   # g/L
    V: float = 10.0               # L
    temperature: float = 37.0     # °C
    pH: float = 7.0

@dataclass
class OrganismConfig:
    name: str = "E. coli K-12"
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    substrates: List[SubstrateConfig] = field(default_factory=list)
    products: List[ProductConfig] = field(default_factory=list)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    oxygen: OxygenConfig = field(default_factory=OxygenConfig)
    heat: HeatConfig = field(default_factory=HeatConfig)
    k_d_max: float = 0.01         # h⁻¹
    k_lysis: float = 0.005        # h⁻¹

@dataclass
class SimulationConfig:
    organism: OrganismConfig = field(default_factory=OrganismConfig)
    bioreactor: BioreactorConfig = field(default_factory=BioreactorConfig)
    initial: InitialConditions = field(default_factory=InitialConditions)
    t_end: float = 100.0          # hours
    dt_output: float = 0.5        # output resolution (hours)


# ======================================================================
# Helper Functions: Rate Modifiers
# ======================================================================

def temp_factor_cardinal(T: float, T_min: float, T_opt: float, T_max: float) -> float:
    """Cardinal temperature model: gamma(T) ∈ [0, 1]"""
    if T <= T_min or T >= T_max:
        return 0.0
    if T <= T_opt:
        return (T - T_min)**2 / ((T_opt - T_min)**2) if T_opt != T_min else 1.0
    else:
        return (T_max - T)**2 / ((T_max - T_opt)**2) if T_max != T_opt else 1.0

def temp_factor_arrhenius(T: float, E_a: float, T_ref: float = 310.15) -> float:
    """Arrhenius temperature factor (T in Kelvin)"""
    R = 0.008314  # kJ/mol/K
    T_k = T + 273.15
    T_ref_k = T_ref
    return np.exp(-E_a / R * (1/T_k - 1/T_ref_k))

def ph_factor_gaussian(pH: float, pH_opt: float, sigma: float) -> float:
    """Gaussian pH factor"""
    return np.exp(-((pH - pH_opt) / sigma)**2)

def ph_factor_cardinal(pH: float, pH_min: float, pH_opt: float, pH_max: float) -> float:
    """Cardinal pH model"""
    if pH <= pH_min or pH >= pH_max:
        return 0.0
    if pH <= pH_opt:
        return (pH - pH_min)**2 / ((pH_opt - pH_min)**2) if pH_opt != pH_min else 1.0
    else:
        return (pH_max - pH)**2 / ((pH_max - pH_opt)**2) if pH_max != pH_opt else 1.0


# ======================================================================
# Core: Specific Rate Functions
# ======================================================================

def specific_growth_rate(S: float, P: float, B: float, T: float, pH: float,
                         growth: GrowthConfig, env: EnvironmentConfig) -> float:
    """Calculate μ (specific growth rate) based on configured model"""
    
    # Substrate-limited growth
    if growth.model == "monod":
        mu_S = S / (growth.K_s + S)
    elif growth.model == "moser":
        mu_S = S**growth.n / (growth.K_s + S**growth.n)
    elif growth.model == "andrews":  # substrate inhibition
        mu_S = S / (growth.K_s + S + S**2 / growth.K_i) if growth.K_i else S / (growth.K_s + S)
    elif growth.model == "haldane":
        mu_S = 1.0 / (1.0 + growth.K_s / S + S / growth.K_i) if growth.K_i else S / (growth.K_s + S)
    elif growth.model == "logistic":
        mu_S = 1.0  # handled separately in dX
    else:
        mu_S = S / (growth.K_s + S)
    
    # Product inhibition
    if growth.model == "aiba" and growth.K_p:
        mu_P = np.exp(-B / growth.K_p)  # using B as inhibitor
    elif growth.K_p and growth.K_p > 0:
        mu_P = 1.0 / (1.0 + B / growth.K_p)
    else:
        mu_P = 1.0
    
    # Temperature factor
    if env.temp_model == "arrhenius":
        f_T = temp_factor_arrhenius(T, env.E_a)
    else:
        f_T = temp_factor_cardinal(T, env.T_min, env.T_opt, env.T_max)
    
    # pH factor
    f_pH = ph_factor_gaussian(pH, env.pH_opt, env.pH_sigma)
    
    return growth.mu_max * mu_S * mu_P * f_T * f_pH


def specific_substrate_uptake(mu: float, S: float, growth: GrowthConfig, 
                               substrate: SubstrateConfig) -> float:
    """Calculate q_S (specific substrate uptake rate)"""
    # From Pirt equation: q_S = μ/Y_XS + m_s
    q_S_maintenance = mu / substrate.Y_XS + growth.m_s
    
    # Uptake kinetics (Michaelis-Menten-like)
    q_S_kinetic = substrate.q_S_max * S / (substrate.K_m + S)
    
    return min(q_S_kinetic, q_S_maintenance) if q_S_kinetic > 0 else q_S_maintenance


def specific_product_formation(mu: float, q_S: float, product: ProductConfig) -> float:
    """Calculate q_P (specific product formation rate) using Luedeking-Piret"""
    # Luedeking-Piret: q_P = α*μ + β
    q_P_lp = product.alpha * mu + product.beta
    
    # Overflow metabolism if applicable
    if product.q_S_crit and product.q_S_crit > 0:
        q_S_overflow = max(0, q_S - product.q_S_crit)
        q_P_overflow = product.Y_PS * q_S_overflow
        q_P_lp += q_P_overflow
    
    return q_P_lp


def specific_oxygen_uptake(mu: float, oxygen: OxygenConfig) -> float:
    """Calculate q_O2 (specific oxygen uptake rate)"""
    return mu / oxygen.Y_XO + oxygen.m_O


# ======================================================================
# ODE System
# ======================================================================

class BioreactorODE:
    """ODE system for a flexible bioreactor model"""
    
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.org = config.organism
        self.br = config.bioreactor
        self.growth = config.organism.growth
        self.env = config.organism.environment
        self.oxy = config.organism.oxygen
        self.heat = config.organism.heat
        
        # Sort substrates by priority
        self.substrates = sorted(config.organism.substrates, key=lambda s: s.priority)
        self.products = config.organism.products
        
        # Feed events
        self.feed_events = sorted(config.bioreactor.feed_profile or [], key=lambda f: f[0])
        self.feed_idx = 0
        
        self.t_last_feed = 0.0
        self.S_feed_accum = 0.0
    
    def _get_feed_rate(self, t: float) -> Tuple[float, float]:
        """Get feed flow rate and substrate concentration at time t"""
        F = 0.0
        S_feed = 0.0
        
        # Check feed events
        while self.feed_idx < len(self.feed_events) and self.feed_events[self.feed_idx][0] <= t:
            event_t, event_F, event_S = self.feed_events[self.feed_idx]
            if self.br.mode == "fed_batch":
                F = event_F  # constant feed after trigger
                S_feed = event_S
            else:
                F = event_F  # instantaneous
                S_feed = event_S
            self.feed_idx += 1
        
        # Chemostat: F = D * V
        # (simplified — could add dilution rate parameter)
        
        return F, S_feed
    
    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        State vector y:
        [0] X_v     - viable biomass (g/L)
        [1] X_d     - dead biomass (g/L)
        [2] S       - substrate (g/L)
        [3] P       - product (g/L)
        [4] B       - byproduct (g/L)
        [5] DO      - dissolved oxygen (g/L)
        [6] V       - volume (L)
        [7] T       - temperature (°C)
        [8] pH      - pH
        """
        X_v, X_d, S, P, B, DO, V, T, pH = y
        
        # Bounds enforcement
        X_v = max(0, X_v)
        X_d = max(0, X_d)
        S = max(0, S)
        DO = max(0, DO)
        V = max(0.01, V)
        
        # Pick active substrate (highest priority with S > 0)
        active_sub = self.substrates[0]
        for sub in self.substrates:
            if sub.name == self.growth.model.split('_')[0]:  # simplified
                active_sub = sub
                break
        # Use first substrate config
        active_sub = self.substrates[0] if self.substrates else SubstrateConfig()
        
        # Specific rates
        mu = specific_growth_rate(S, P, B, T, pH, self.growth, self.env)
        
        # Logistic growth uses X_max
        if self.growth.model == "logistic":
            mu = self.growth.mu_max * (1 - X_v / self.growth.X_max)
        
        q_S = specific_substrate_uptake(mu, S, self.growth, active_sub)
        q_O2 = specific_oxygen_uptake(mu, self.oxy) if self.oxy.q_O2_max > 0 else 0
        
        # Product/byproduct formation
        q_P_total = 0.0
        q_B_total = 0.0
        K_p_inhibition = 1.0
        
        for prod in self.products:
            q_prod = specific_product_formation(mu, q_S, prod)
            if prod.product_type == "product":
                q_P_total += q_prod
            else:
                q_B_total += q_prod
            
            # Product inhibition factor
            if prod.inhibition_type == "aiba" and prod.K_p:
                inhibitor = P if prod.product_type == "product" else B
                K_p_inhibition *= np.exp(-inhibitor / prod.K_p)
            elif prod.inhibition_type == "linear" and prod.K_p:
                inhibitor = P if prod.product_type == "product" else B
                K_p_inhibition *= 1.0 / (1.0 + inhibitor / prod.K_p)
        
        # Apply product inhibition to growth
        mu *= K_p_inhibition
        
        # Death rate
        k_d = self.org.k_d_max * (1.0 - S / (self.growth.K_s + S))  # increases at low S
        k_d = min(k_d, self.org.k_d_max * 5)  # cap at 5x
        
        # Feed/flow
        F, S_feed = self._get_feed_rate(t)
        F_out = 0.0
        
        if self.br.mode == "chemostat":
            D = F / V if V > 0 else 0
            F_out = F  # steady state volume
        elif self.br.mode == "fed_batch":
            D = F / V if V > 0 else 0
            F_out = 0
        else:  # batch
            D = 0
            F_out = 0
        
        # ===== ODEs =====
        
        # 1. Viable biomass: dX_v/dt = (μ - k_d - D)*X_v
        dXv = (mu - k_d) * X_v - D * X_v
        
        # 2. Dead biomass: dX_d/dt = k_d*X_v - k_lysis*X_d - D*X_d
        dXd = k_d * X_v - self.org.k_lysis * X_d - D * X_d
        
        # 3. Substrate: dS/dt = -q_S*X_v + D*(S_feed - S)
        dS = -q_S * X_v + D * (S_feed - S) if D > 0 else -q_S * X_v
        
        # 4. Product: dP/dt = q_P*X_v - D*P
        dP = q_P_total * X_v - D * P
        
        # 5. Byproduct: dB/dt = q_B*X_v - D*B
        dB = q_B_total * X_v - D * B
        
        # 6. Dissolved oxygen: dDO/dt = kLa*(O2* - DO) - q_O2*X_v
        dDO = self.br.kLa * (self.br.O2_star - DO) - q_O2 * X_v
        
        # 7. Volume: dV/dt = F - F_out
        dV = F - F_out
        
        # 8. Temperature (simplified energy balance)
        Q_growth = self.heat.Y_QX * mu * X_v * V
        Q_substrate = self.heat.Y_QS * q_S * X_v * V
        Q_cooling = self.br.heat_transfer * (T - self.br.T_setpoint)
        dT = (Q_growth + Q_substrate - Q_cooling) / (V * 4.18)  # 4.18 kJ/L/K for water
        
        # 9. pH (simplified: organic acid accumulation lowers pH)
        dpH = -0.01 * q_B_total * X_v  # empirical
        
        return np.array([dXv, dXd, dS, dP, dB, dDO, dV, dT, dpH])


# ======================================================================
# Simulation Runner
# ======================================================================

def run_simulation(config: SimulationConfig) -> Dict:
    """Run bioreactor simulation and return results"""
    
    ode = BioreactorODE(config)
    
    # Initial state
    ic = config.initial
    y0 = np.array([
        ic.X_v, ic.X_d, ic.S, ic.P, ic.B,
        ic.dissolved_O2, ic.V, ic.temperature, ic.pH
    ])
    
    # Solve ODE
    t_span = (0, config.t_end)
    t_eval = np.arange(0, config.t_end + config.dt_output, config.dt_output)
    
    sol = solve_ivp(
        ode, t_span, y0,
        method='LSODA',
        t_eval=t_eval,
        max_step=1.0,
        rtol=1e-6, atol=1e-8
    )
    
    # Extract results
    X_v, X_d, S, P, B, DO, V, T, pH = sol.y
    
    # Compute derived quantities
    mu = np.zeros_like(sol.t)
    q_S = np.zeros_like(sol.t)
    q_P = np.zeros_like(sol.t)
    q_B = np.zeros_like(sol.t)
    viability = np.zeros_like(sol.t)
    
    total_biomass = X_v + X_d
    viability = np.where(total_biomass > 0, X_v / total_biomass, 0)
    
    return {
        'time': sol.t.tolist(),
        'X_v': X_v.tolist(),
        'X_d': X_d.tolist(),
        'total_biomass': total_biomass.tolist(),
        'substrate': S.tolist(),
        'product': P.tolist(),
        'byproduct': B.tolist(),
        'dissolved_oxygen': DO.tolist(),
        'volume': V.tolist(),
        'temperature': T.tolist(),
        'pH': pH.tolist(),
        'viability': viability.tolist(),
        'status': 'success' if sol.success else 'failed',
        'message': sol.message,
        't_events': len(sol.t_events) if sol.t_events else 0,
        'nfev': sol.nfev
    }


# ======================================================================
# Example: E. coli K-12 Batch Fermentation
# ======================================================================

def create_ecoli_batch_example() -> SimulationConfig:
    """Create a simulation config for E. coli K-12 batch fermentation"""
    return SimulationConfig(
        organism=OrganismConfig(
            name="E. coli K-12 MG1655",
            growth=GrowthConfig(
                model="monod",
                mu_max=0.70,
                K_s=0.02,
                Y_XS=0.50,
                m_s=0.025
            ),
            substrates=[
                SubstrateConfig(
                    name="glucose",
                    q_S_max=1.50,
                    K_m=0.01,
                    Y_XS=0.50,
                    priority=1
                )
            ],
            products=[
                ProductConfig(
                    name="acetate",
                    product_type="byproduct",
                    model="luedeking-piret",
                    alpha=0.0,
                    beta=0.02,
                    Y_PS=0.15,
                    q_S_crit=0.80,
                    inhibition_type="linear",
                    K_p=10.0
                )
            ],
            environment=EnvironmentConfig(
                T_min=8.0, T_opt=37.0, T_max=45.0,
                pH_min=5.0, pH_opt=7.0, pH_max=9.0, pH_sigma=0.8
            ),
            oxygen=OxygenConfig(q_O2_max=0.40, Y_XO=0.95, m_O=0.01),
            heat=HeatConfig(Y_QX=19.0, Y_QS=14.0),
            k_d_max=0.01,
            k_lysis=0.005
        ),
        bioreactor=BioreactorConfig(
            mode="batch",
            V_initial=10.0,
            V_max=20.0,
            kLa=150.0,
            T_setpoint=37.0,
            pH_setpoint=7.0
        ),
        initial=InitialConditions(
            X_v=1.0,
            X_d=0.0,
            S=5.0,
            P=0.0,
            B=0.0,
            dissolved_O2=0.007,
            V=10.0,
            temperature=37.0,
            pH=7.0
        ),
        t_end=24.0,
        dt_output=0.1
    )


if __name__ == "__main__":
    # Quick test
    config = create_ecoli_batch_example()
    results = run_simulation(config)
    
    print(f"Status: {results['status']}")
    print(f"Message: {results['message']}")
    print(f"Function evaluations: {results['nfev']}")
    print(f"Time points: {len(results['time'])}")
    print(f"Final biomass (X_v): {results['X_v'][-1]:.2f} g/L")
    print(f"Final substrate: {results['substrate'][-1]:.3f} g/L")
    print(f"Final byproduct: {results['byproduct'][-1]:.3f} g/L")
    print(f"Final viability: {results['viability'][-1]:.3f}")
    
    # Save as JSON
    with open('/home/renku/work/SDSC-STEM4Health-group10/bioreactor-app/backend/python_sim/example_output.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to example_output.json")

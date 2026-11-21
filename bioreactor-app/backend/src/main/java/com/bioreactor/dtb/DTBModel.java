package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.hipparchus.ode.*;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;

import java.util.*;

/**
 * DTB-specific process model for the hackathon challenge.
 *
 * Ported from Python: src/mechanistic.py (team's reference model on main) and
 * parameter_estimation.custom_ode.
 *
 * Key differences from the textbook MonodModel:
 * - Time is in days.
 * - Cells, byproduct, and product do not enter with the feed.
 * - Feed concentrations for each stream are fitted parameters.
 * - Volume is set to the measured volume at each sampling time.
 */
public class DTBModel implements ProcessModel {

    @Override
    public List<ModelParameter> parameters() {
        return List.of(
            new ModelParameter("mu_max",   0.70,  0.001, 5.0,   "1/d",  "Maximum specific growth rate"),
            new ModelParameter("K_s",      0.02,  0.0001, 5.0,  "g/L",  "Substrate half-saturation"),
            new ModelParameter("Y_XS",     0.50,  0.01,  0.95,  "g/g",  "Biomass yield on substrate"),
            new ModelParameter("m_s",      0.025, 0.0,   0.5,   "g/g/d","Maintenance coefficient"),
            new ModelParameter("alpha",    0.0,   0.0,   1.0,   "g/g",  "Growth-associated byproduct"),
            new ModelParameter("beta",     0.005, 0.0,   0.1,   "1/d",  "Non-growth byproduct rate"),
            new ModelParameter("k_d",      0.005, 0.0,   0.1,   "1/d",  "Maximum death rate"),
            new ModelParameter("k_lysis",  0.001, 0.0,   0.05,  "1/d",  "Lysis rate"),
            new ModelParameter("feedB_S",  500.0, 10.0, 1000.0, "g/L",  "Feed B substrate concentration"),
            new ModelParameter("feedC_S",  500.0, 10.0, 1000.0, "g/L",  "Feed C substrate concentration"),
            new ModelParameter("feedD_S",  500.0, 10.0, 1000.0, "g/L",  "Feed D substrate concentration")
        );
    }

    @Override
    public double[] rates(BioreactorState state, Map<String, Double> params) {
        double mu_max  = params.getOrDefault("mu_max", 0.70);
        double K_s     = params.getOrDefault("K_s", 0.02);
        double Y_XS    = params.getOrDefault("Y_XS", 0.50);
        double m_s     = params.getOrDefault("m_s", 0.025);
        double alpha   = params.getOrDefault("alpha", 0.0);
        double beta    = params.getOrDefault("beta", 0.005);
        double k_d_val = params.getOrDefault("k_d", 0.005);
        double k_lys   = params.getOrDefault("k_lysis", 0.001);

        double xv = Math.max(0, state.xv());
        double xd = Math.max(0, state.xd());
        double substrate = Math.max(0, state.substrate());

        // Monod: μ = μ_max * S / (K_s + S)
        double mu = mu_max * substrate / (K_s + substrate);

        // Pirt: q_S = μ / Y_XS + m_s
        double q_S = (Y_XS > 0 ? mu / Y_XS : 0) + m_s;

        // Luedeking-Piret: byproduct from overflow
        double q_B = alpha * mu + beta;

        // Death: function of growth rate
        double k_d_actual = k_d_val * Math.max(0, 1.0 - mu / Math.max(mu_max, 1e-9));
        double dXv = (mu - k_d_actual) * xv;
        double dXd = k_d_actual * xv - k_lys * xd;

        // Substrate consumption
        double dS = -q_S * xv;

        return new double[]{dXv, dXd, dS, 0.0, q_B * xv,
                            0.0, 0.0, 0.0, 0.0};
    }

    @Override
    public Map<String, Double> feedConcentrations(Map<String, Double> params) {
        return Map.of(
            "substrate", params.getOrDefault("feedB_S", 500.0)
        );
    }
}

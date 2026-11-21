package com.bioreactor.dtb;

import com.bioreactor.engine.*;

import java.util.*;

/**
 * Advanced DTB model — v4 — incorporating full improvements from brainstorming.
 *
 * IMPROVEMENTS:
 *  1. THRESHOLD product inhibition (Pirog-style):
 *     No inhibition at all when B ≤ B_thresh.
 *     Linear decline from 1.0 → 0.0 between B_thresh and B_crit.
 *
 *  2. BYPRODUCT CONSUMPTION as secondary substrate:
 *     Data shows byproduct levels DROP in late phase (2.3 → 1.3 g/L).
 *     q_B_consume = q_B_max · B/(K_Bp+B) · K_sB/(K_sB+S)
 *     Only consumed when primary substrate S is low.
 *     Consumed byproduct supports growth via Y_XB yield.
 *
 *  3. Haldane substrate inhibition:
 *     μ = μ_max · S/(K_s + S + S²/K_i)
 *
 *  4. Death tied to substrate (not μ/μ_max):
 *     k_d = k_d_max · (1 - S/(K_s+S))
 *
 *  5. Feeds C & D treated as NON-NUTRITIVE (pH control, trace supplement).
 *     Only Feed B carries substrate. Confirmed by data: C ≈ 0.0008 L/h, D ≈ 0.016 L/h.
 */
public class DTBModelAdvanced implements ProcessModel {

    @Override
    public List<ModelParameter> parameters() {
        return List.of(
            // Growth (Monod + Haldane)
            new ModelParameter("mu_max",   0.70,  0.001, 2.0,   "1/d",  "Maximum specific growth rate"),
            new ModelParameter("K_s",      0.02,  0.0001, 2.0,  "g/L",  "Substrate half-saturation (Monod)"),
            new ModelParameter("K_i",      50.0,  1.0,   200.0, "g/L",  "Haldane substrate inhibition"),

            // Threshold product inhibition
            new ModelParameter("B_thresh", 1.0,   0.1,   3.0,   "g/L",  "Byproduct threshold for inhibition onset"),
            new ModelParameter("B_crit",   5.0,   1.0,   10.0,  "g/L",  "Byproduct at which growth stops completely"),

            // Yield & maintenance
            new ModelParameter("Y_XS",     0.50,  0.05,  0.90,  "g/g",  "Biomass yield on substrate"),
            new ModelParameter("m_s",      0.025, 0.0,   0.2,   "g/g/d","Maintenance coefficient"),

            // Byproduct PRODUCTION (Luedeking-Piret)
            new ModelParameter("alpha",    0.0,   0.0,   0.5,   "g/g",  "Growth-associated byproduct formation"),
            new ModelParameter("beta",     0.01,  0.0,   0.1,   "1/d",  "Non-growth-associated byproduct rate"),

            // Byproduct CONSUMPTION
            new ModelParameter("q_B_max",  0.1,   0.0,   0.5,   "g/g/d","Max byproduct consumption rate"),
            new ModelParameter("K_Bp",     1.0,   0.1,   5.0,   "g/L",  "Byproduct half-sat for consumption"),
            new ModelParameter("K_sB",     0.5,   0.01,  3.0,   "g/L",  "Substrate level below which byproduct consumed"),
            new ModelParameter("Y_XB",     0.40,  0.1,   0.80,  "g/g",  "Biomass yield from byproduct"),

            // Death & lysis
            new ModelParameter("k_d",      0.01,  0.0,   0.1,   "1/d",  "Maximum death rate"),
            new ModelParameter("k_lysis",  0.005, 0.0,   0.05,  "1/d",  "Lysis rate"),

            // Feed B concentration (only nutrient feed)
            new ModelParameter("feedB_S",  120.0, 10.0,  500.0, "g/L",  "Feed B substrate concentration")
        );
    }

    @Override
    public double[] rates(BioreactorState state, Map<String, Double> params) {
        double mu_max   = params.getOrDefault("mu_max", 0.70);
        double K_s      = params.getOrDefault("K_s", 0.02);
        double K_i      = params.getOrDefault("K_i", 50.0);
        double B_thresh = params.getOrDefault("B_thresh", 1.0);
        double B_crit   = params.getOrDefault("B_crit", 5.0);
        double Y_XS     = params.getOrDefault("Y_XS", 0.50);
        double m_s      = params.getOrDefault("m_s", 0.025);
        double alpha    = params.getOrDefault("alpha", 0.0);
        double beta     = params.getOrDefault("beta", 0.01);
        double q_B_max  = params.getOrDefault("q_B_max", 0.1);
        double K_Bp     = params.getOrDefault("K_Bp", 1.0);
        double K_sB     = params.getOrDefault("K_sB", 0.5);
        double Y_XB     = params.getOrDefault("Y_XB", 0.40);
        double k_d_val  = params.getOrDefault("k_d", 0.01);
        double k_lys    = params.getOrDefault("k_lysis", 0.005);

        double xv = Math.max(0, state.xv());
        double xd = Math.max(0, state.xd());
        double S  = Math.max(0, state.substrate());
        double B  = Math.max(0, state.byproduct());

        // === Haldane growth ===
        double mu_S = S / (K_s + S + (S * S) / Math.max(K_i, 0.01));
        double mu_base = mu_max * mu_S;

        // === THRESHOLD product inhibition ===
        double mu;
        if (B <= B_thresh) {
            mu = mu_base;           // no inhibition
        } else if (B >= B_crit) {
            mu = 0.0;               // growth stops
        } else {
            double frac = (B - B_thresh) / (B_crit - B_thresh);
            mu = mu_base * (1.0 - frac);
        }

        // === Byproduct PRODUCTION ===
        double q_B_prod = alpha * mu + beta;

        // === Byproduct CONSUMPTION (when S is low) ===
        double sInhibition = K_sB / (K_sB + S);      // ~1 when S≈0
        double bDrive = B / (K_Bp + B);               // Michaelis for B
        double q_B_consume = q_B_max * bDrive * sInhibition;
        double q_B_net = q_B_prod - q_B_consume;

        // === Growth from byproduct consumption ===
        double mu_from_byproduct = q_B_consume * Y_XB;

        // === Substrate uptake ===
        double Y_BS = 0.70;  // fixed: g byproduct per g substrate (overflow stoichiometry)
        double q_S = (Y_XS > 0 ? mu / Y_XS : 0) + m_s + (Y_BS > 0 ? q_B_prod / Y_BS : 0);

        // === Death ===
        double substFactor = Math.max(0, 1.0 - S / (S + K_s));
        double k_d_actual = k_d_val * substFactor;

        // === Mass balances ===
        double dXv = (mu + mu_from_byproduct - k_d_actual) * xv;
        double dXd = k_d_actual * xv - k_lys * xd;
        double dS  = -q_S * xv;
        double dB  = q_B_net * xv;

        return new double[]{dXv, dXd, dS, 0.0, dB, 0.0, 0.0, 0.0, 0.0};
    }

    @Override
    public Map<String, Double> feedConcentrations(Map<String, Double> params) {
        return Map.of("substrate", params.getOrDefault("feedB_S", 120.0));
    }
}

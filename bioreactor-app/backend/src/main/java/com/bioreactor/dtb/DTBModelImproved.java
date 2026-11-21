package com.bioreactor.dtb;

import com.bioreactor.engine.*;

import java.util.*;

/**
 * Improved DTB model incorporating insights from:
 *  - T.P. Pirog's microbiology textbook (multi-substrate limitation, product inhibition thresholds)
 *  - The team's Python v3 (Haldane substrate inhibition, proper feed mass balance)
 *  - Literature on industrial fermentation modeling
 *
 * Key improvements over the original DTBModel:
 *
 *  1. Haldane/Andrews substrate inhibition:
 *     μ = μ_max · S / (K_s + S + S²/K_i)
 *     Prevents over-prediction of growth at high substrate after bolus feeds.
 *
 *  2. Product inhibition (Jerusalimsky hyperbolic):
 *     μ_eff = μ · K_p / (K_p + B)
 *     Accumulated byproducts slow growth, improving late-phase predictions.
 *
 *  3. Death rate tied to substrate (not μ/μ_max):
 *     k_d = k_d_max · (1 - S/(K_s + S))
 *     Biologically correct: death increases as substrate is depleted.
 *
 *  4. Substrate mass balance for byproduct formation:
 *     q_S = μ/Y_XS + m_s + q_B/Y_BS
 *     Byproduct synthesis consumes substrate.
 *
 *  5. Temperature & pH environmental factors restored:
 *     Growth is modified by f(T) and f(pH) from the textbook MonodModel.
 */
public class DTBModelImproved implements ProcessModel {

    // Environmental constants (same as MonodModel)
    private static final double T_min = 8.0, T_opt = 37.0, T_max = 45.0;
    private static final double pH_opt = 7.0, pH_sigma = 0.8;

    @Override
    public List<ModelParameter> parameters() {
        return List.of(
            // Growth
            new ModelParameter("mu_max",   0.70,  0.001, 5.0,   "1/d",  "Maximum specific growth rate"),
            new ModelParameter("K_s",      0.02,  0.0001, 5.0,  "g/L",  "Substrate half-saturation"),
            new ModelParameter("K_i",      50.0,  0.1,   200.0, "g/L",  "Haldane substrate inhibition constant"),

            // Yield & maintenance
            new ModelParameter("Y_XS",     0.50,  0.01,  0.95,  "g/g",  "Biomass yield on substrate"),
            new ModelParameter("m_s",      0.025, 0.0,   0.5,   "g/g/d","Maintenance coefficient"),
            new ModelParameter("Y_BS",     0.90,  0.1,   2.0,   "g/g",  "Byproduct yield from substrate"),

            // Byproduct (Luedeking-Piret)
            new ModelParameter("alpha",    0.0,   0.0,   1.0,   "g/g",  "Growth-associated byproduct"),
            new ModelParameter("beta",     0.005, 0.0,   0.1,   "1/d",  "Non-growth byproduct rate"),

            // Product inhibition
            new ModelParameter("K_p",      50.0,  1.0,   200.0, "g/L",  "Product inhibition constant"),

            // Death & lysis
            new ModelParameter("k_d",      0.005, 0.0,   0.1,   "1/d",  "Maximum death rate"),
            new ModelParameter("k_lysis",  0.001, 0.0,   0.05,  "1/d",  "Lysis rate"),

            // Feed concentrations
            new ModelParameter("feedB_S",  80.0,  10.0,  400.0, "g/L",  "Feed B substrate concentration"),
            new ModelParameter("feedC_S",  0.0,   0.0,   100.0, "g/L",  "Feed C substrate concentration"),
            new ModelParameter("feedD_S",  50.0,  0.0,   200.0, "g/L",  "Feed D substrate concentration")
        );
    }

    @Override
    public double[] rates(BioreactorState state, Map<String, Double> params) {
        // --- Extract parameters ---
        double mu_max  = params.getOrDefault("mu_max", 0.70);
        double K_s     = params.getOrDefault("K_s", 0.02);
        double K_i     = params.getOrDefault("K_i", 50.0);
        double Y_XS    = params.getOrDefault("Y_XS", 0.50);
        double m_s     = params.getOrDefault("m_s", 0.025);
        double Y_BS    = params.getOrDefault("Y_BS", 0.90);
        double alpha   = params.getOrDefault("alpha", 0.0);
        double beta    = params.getOrDefault("beta", 0.005);
        double K_p     = params.getOrDefault("K_p", 50.0);
        double k_d_val = params.getOrDefault("k_d", 0.005);
        double k_lys   = params.getOrDefault("k_lysis", 0.001);

        // --- State variables (clamped to non-negative) ---
        double xv = Math.max(0, state.xv());
        double xd = Math.max(0, state.xd());
        double substrate = Math.max(0, state.substrate());
        double byproduct = Math.max(0, state.byproduct());

        // --- IMPROVEMENT 5: Environment factors ---
        double f_T = EnvironmentFactors.tempFactorCardinal(state.temperature(), T_min, T_opt, T_max);
        double f_pH = EnvironmentFactors.phFactorGaussian(state.pH(), pH_opt, pH_sigma);

        // --- IMPROVEMENT 1: Haldane substrate inhibition ---
        // μ = μ_max · S / (K_s + S + S²/K_i)  ·  f_T · f_pH
        double mu_S = substrate / (K_s + substrate + (substrate * substrate) / Math.max(K_i, 0.01));
        double mu = mu_max * mu_S * f_T * f_pH;

        // --- IMPROVEMENT 2: Product inhibition (Jerusalimsky hyperbolic) ---
        // μ_eff = μ · K_p / (K_p + B)
        double inhibition = K_p / (K_p + byproduct);
        double mu_eff = mu * inhibition;

        // --- Luedeking-Piret byproduct ---
        double q_B = alpha * mu_eff + beta;

        // --- IMPROVEMENT 4: Substrate uptake with byproduct mass balance ---
        // q_S = μ_eff / Y_XS + m_s + q_B / Y_BS
        double q_S = (Y_XS > 0 ? mu_eff / Y_XS : 0) + m_s + (Y_BS > 0 ? q_B / Y_BS : 0);

        // --- IMPROVEMENT 3: Death rate tied to substrate ---
        // k_d = k_d_max · (1 - S / (K_s + S))
        double substrateFactor = Math.max(0, 1.0 - substrate / (substrate + K_s));
        double k_d_actual = k_d_val * substrateFactor;

        double dXv = (mu_eff - k_d_actual) * xv;
        double dXd = k_d_actual * xv - k_lys * xd;

        // Substrate consumption
        double dS = -q_S * xv;

        // Byproduct production
        double dB = q_B * xv;

        // No primary product, oxygen, volume, temperature, pH dynamics
        // (these are controlled/negligible for DTB challenge)
        return new double[]{dXv, dXd, dS, 0.0, dB,
                            0.0, 0.0, 0.0, 0.0};
    }

    @Override
    public Map<String, Double> feedConcentrations(Map<String, Double> params) {
        return Map.of(
            "substrate", params.getOrDefault("feedB_S", 80.0)
        );
    }
}

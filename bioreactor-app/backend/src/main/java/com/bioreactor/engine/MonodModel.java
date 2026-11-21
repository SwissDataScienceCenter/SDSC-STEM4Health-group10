package com.bioreactor.engine;

import java.util.*;

/**
 * Textbook microbial growth model: Monod + Pirt + Luedeking-Piret.
 *
 * Ported from Python: simulation_engine.specific_growth_rate (monod),
 * simulation_engine.specific_substrate_uptake (Pirt),
 * simulation_engine.specific_product_formation (Luedeking-Piret).
 */
public class MonodModel implements ProcessModel {

    // Fixed values from the OrganismConfig defaults
    private static final double T_min = 8.0, T_opt = 37.0, T_max = 45.0;
    private static final double pH_opt = 7.0, pH_sigma = 0.8;
    private static final double k_d_max = 0.01;
    private static final double k_lysis = 0.005;
    private static final double kLa = 150.0;
    private static final double O2_star = 0.007;
    private static final double Y_XO = 0.95;
    private static final double m_O = 0.01;
    private static final double Y_QX = 19.0;
    private static final double Y_QS = 14.0;
    private static final double heat_transfer = 0.5;
    private static final double T_setpoint = 37.0;

    @Override
    public List<ModelParameter> parameters() {
        return List.of(
            new ModelParameter("mu_max",    0.70,  0.01, 5.0,  "1/h",  "Maximum specific growth rate"),
            new ModelParameter("K_s",       0.02,  0.001, 5.0, "g/L",  "Substrate affinity constant"),
            new ModelParameter("Y_XS",      0.50,  0.01, 0.95, "g/g",  "Biomass yield on substrate"),
            new ModelParameter("m_s",       0.025, 0.0,  0.5,  "g/g/h","Maintenance coefficient"),
            new ModelParameter("alpha",     0.0,   0.0,  1.0,  "g/g",  "Growth-associated product"),
            new ModelParameter("beta",      0.02,  0.0,  0.5,  "1/h",  "Non-growth product rate"),
            new ModelParameter("k_d",       0.01,  0.0,  0.1,  "1/h",  "Maximum death rate"),
            new ModelParameter("k_lysis",   0.005, 0.0,  0.05, "1/h",  "Lysis rate")
        );
    }

    @Override
    public double[] rates(BioreactorState s, Map<String, Double> p) {
        double mu_max  = p.getOrDefault("mu_max", 0.70);
        double K_s     = p.getOrDefault("K_s", 0.02);
        double Y_XS    = p.getOrDefault("Y_XS", 0.50);
        double m_s     = p.getOrDefault("m_s", 0.025);
        double alpha   = p.getOrDefault("alpha", 0.0);
        double beta    = p.getOrDefault("beta", 0.02);
        double k_d_val = p.getOrDefault("k_d", 0.01);
        double k_lysis_val = p.getOrDefault("k_lysis", 0.005);

        double xv = Math.max(0, s.xv());
        double xd = Math.max(0, s.xd());
        double substrate = Math.max(0, s.substrate());
        double V = Math.max(0.01, s.volume());
        double DO = Math.max(0, s.dissolvedO2());

        // Environment factors
        double f_T = EnvironmentFactors.tempFactorCardinal(s.temperature(), T_min, T_opt, T_max);
        double f_pH = EnvironmentFactors.phFactorGaussian(s.pH(), pH_opt, pH_sigma);

        // Monod growth: mu = mu_max * S/(K_s+S) * f_T * f_pH
        double mu = mu_max * (substrate / (K_s + substrate)) * f_T * f_pH;

        // Pirt substrate uptake: q_S = mu / Y_XS + m_s
        double q_S = (Y_XS > 0 ? mu / Y_XS : 0) + m_s;

        // Luedeking-Piret product/byproduct: q_B = alpha*mu + beta
        double q_B = alpha * mu + beta;

        // Death + lysis
        double k_d_actual = k_d_val * Math.max(0, 1.0 - substrate / (substrate + K_s));
        double dXv = (mu - k_d_actual) * xv;
        double dXd = k_d_actual * xv - k_lysis_val * xd;

        // Substrate: -q_S * X_v
        double dS = -q_S * xv;

        // No primary product in this model
        double dP = 0.0;
        double dB = q_B * xv;

        // Oxygen
        double q_O2 = (mu_max > 0 ? mu / Y_XO + m_O : m_O); // simpler than Python which scaled by mu/mu_max
        double dDO = kLa * (O2_star - DO) - q_O2 * xv;

        // Volume (no feed: dV=0)
        double dV = 0.0;

        // Temperature (energy balance, simplified)
        double Q_growth = Y_QX * mu * xv * V;
        double Q_sub = Y_QS * q_S * xv * V;
        double Q_cool = heat_transfer * (s.temperature() - T_setpoint);
        double dT = (Q_growth + Q_sub - Q_cool) / (V * 4.18);

        // pH
        double dpH = -0.01 * q_B * xv;

        return new double[]{dXv, dXd, dS, dP, dB, dDO, dV, dT, dpH};
    }

    @Override
    public Map<String, Double> feedConcentrations(Map<String, Double> params) {
        return Map.of("substrate", 500.0); // g/L in feed, default
    }
}

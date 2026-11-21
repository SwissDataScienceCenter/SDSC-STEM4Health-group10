package com.bioreactor.engine;

import java.util.*;

/**
 * Configurable bioreactor model that maps organism parameters onto
 * the standard BioreactorState (Xv, Xd, S, P, B, DO, V, T, pH).
 *
 * Multiple substrates and products are supported via hidden states.
 * The primary substrate goes in the S field; primary product in P;
 * primary byproduct in B.
 */
public class ConfigurableModel implements ProcessModel {

    private final OrganismConfig config;
    private final int nHidden;  // additional substrates + products beyond the first of each

    public ConfigurableModel(OrganismConfig config) {
        this.config = config;
        // Currently mapping all products/byproducts onto the standard 9 core fields.
        // Future: hidden states for multi-substrate/product models.
        this.nHidden = 0;
    }

    @Override
    public int hiddenStateCount() { return nHidden; }

    @Override
    public List<ModelParameter> parameters() {
        List<ModelParameter> ps = new ArrayList<>();
        ps.add(new ModelParameter("mu_max", config.muMax,      0.01, 5.0,   "1/h",    "Max specific growth rate"));
        ps.add(new ModelParameter("K_s",    config.kS,         0.001, 10.0,  "g/L",    "Half-saturation constant"));
        ps.add(new ModelParameter("Y_XS",   config.yXS,        0.01,  1.0,   "g/g",    "Biomass yield on substrate"));
        ps.add(new ModelParameter("m_s",    config.mS,         0.001, 0.5,   "g/g/h",  "Maintenance coefficient"));
        if (config.kI != null && config.kI > 0)
            ps.add(new ModelParameter("K_i", config.kI,        1.0, 1000.0, "g/L",     "Substrate inhibition"));
        if (config.xMax != null && config.xMax > 0)
            ps.add(new ModelParameter("X_max", config.xMax,    1.0, 200.0,  "g/L",     "Carrying capacity"));
        ps.add(new ModelParameter("k_d_max", config.kdMax,     0.0,  0.5,   "1/h",    "Max death rate"));
        ps.add(new ModelParameter("k_lysis", config.kLysis,    0.0,  0.5,   "1/h",    "Lysis rate"));
        return ps;
    }

    @Override
    public double[] rates(BioreactorState state, Map<String, Double> p) {
        double mu_max = param(p, "mu_max", config.muMax);
        double K_s    = param(p, "K_s", config.kS);
        double Y_XS   = param(p, "Y_XS", config.yXS);
        double m_s    = param(p, "m_s", config.mS);
        double K_i    = param(p, "K_i", config.kI != null ? config.kI : 1e9);
        double X_max  = param(p, "X_max", config.xMax != null ? config.xMax : 1e9);
        double kd_max = param(p, "k_d_max", config.kdMax);
        double kLysis = param(p, "k_lysis", config.kLysis);

        double Xv = Math.max(0, state.xv());       // viable cells
        double Xd = Math.max(0, state.xd());       // dead cells
        double S  = Math.max(0, state.substrate()); // primary substrate
        double P  = Math.max(0, state.product());   // primary product
        double B  = Math.max(0, state.byproduct()); // primary byproduct
        double DO = Math.max(0, state.dissolvedO2());
        double T  = state.temperature();
        double pHVal = state.pH();

        // ── Growth rate ──
        double mu = mu_max;
        switch (config.growthModel) {
            case "andrews" -> mu *= S / (K_s + S + S * S / K_i);
            case "moser" -> {
                double n = config.n != null ? config.n : 1.0;
                mu *= Math.pow(S, n) / (Math.pow(K_s, n) + Math.pow(S, n));
            }
            default -> mu *= S / (K_s + S);
        }
        // Logistic brake
        mu *= Math.max(0, 1.0 - Xv / X_max);
        // Temperature
        if (config.tMin > 0 && config.tOpt > 0 && config.tMax > 0)
            mu *= cardinalTemp(T);
        // pH
        if (config.pHOpt > 0 && config.pHSigma > 0)
            mu *= Math.exp(-((pHVal - config.pHOpt) * (pHVal - config.pHOpt))
                           / (2.0 * config.pHSigma * config.pHSigma));

        // ── Death ──
        double kd = kd_max * (1.0 - Math.min(1.0, S / (K_s + S))); // death increases as S depletes

        // ── Substrate uptake ──
        double qS = mu / Math.max(1e-6, Y_XS) + m_s;

        // ── Byproduct (from overflow) ──
        double qSCrit = 1.0; // generic overflow threshold
        double overflow = 0;
        if (qS > qSCrit && config.byproducts.size() > 0) {
            var bp = config.byproducts.get(0);
            overflow = bp.yPS() * (qS - qSCrit);
        }
        double dP = 0;
        double dB = overflow * Xv;

        // ── Products (Luedeking-Piret) ──
        if (config.products.size() > 0) {
            var prod = config.products.get(0);
            dP = prod.alpha() * mu * Xv + prod.beta() * Xv;
        }

        // ── O2 consumption (growth-linked) ──
        double qO2 = config.qO2Max != null ? config.qO2Max * mu / mu_max : 0;
        double dDO = -qO2 * Xv;

        // Return rate vector: 9 core fields
        return new double[]{
            (mu - kd) * Xv,                 // dXv
            kd * Xv - kLysis * Xd,          // dXd
            -qS * Xv,                        // dS
            dP,                              // dP
            dB,                              // dByproduct
            Math.min(0, dDO),               // dDO (never negative inflow)
            0,                               // dV (handled by events)
            0,                               // dT (isothermal assumption)
            0                                // dpH
        };
    }

    private double param(Map<String, Double> p, String key, double def) {
        return p.getOrDefault(key, def);
    }

    private double cardinalTemp(double T) {
        if (T <= config.tMin || T >= config.tMax) return 0;
        if (T <= config.tOpt)
            return (T - config.tMin) * (T - config.tMin)
                   / ((config.tOpt - config.tMin) * (config.tOpt - config.tMin));
        else
            return (config.tMax - T) * (config.tMax - T)
                   / ((config.tMax - config.tOpt) * (config.tMax - config.tOpt));
    }

    @Override
    public Map<String, Double> feedConcentrations(Map<String, Double> p) {
        return Map.of("substrate", param(p, "feed_substrate", 250.0));
    }

    @Override
    public Map<String, Double> feedStreamConcentrations(String feedName, Map<String, Double> p) {
        return switch (feedName) {
            case "substrate", "feed_substrate" -> Map.of("substrate", param(p, "feed_substrate", 250.0));
            case "feed_b" -> Map.of("substrate", param(p, "feed_b", 50.0));
            default -> Map.of();
        };
    }

    // ── Config record types ──

    public record OrganismConfig(
        String id, String name,
        String growthModel, double muMax, double kS, Double kI,
        double yXS, double mS, Double xMax, Double n,
        double kdMax, double kLysis,
        List<SubstrateConfig> substrates,
        List<ProductConfig> products,
        List<ProductConfig> byproducts,
        double tMin, double tOpt, double tMax,
        double pHMin, double pHOpt, double pHMax, double pHSigma,
        Double qO2Max, Double yXO, Double mO,
        Double yQX, Double yQS
    ) {
        public OrganismConfig {
            substrates = substrates == null ? List.of() : List.copyOf(substrates);
            products   = products   == null ? List.of() : List.copyOf(products);
            byproducts = byproducts == null ? List.of() : List.copyOf(byproducts);
        }
    }

    public record SubstrateConfig(String id, String name, double qSMax, double kM, int priority) {}
    public record ProductConfig(
        String name, String type, double alpha, double beta, double yPS,
        Double qSCrit, String inhibitionType, Double kP
    ) {}
}

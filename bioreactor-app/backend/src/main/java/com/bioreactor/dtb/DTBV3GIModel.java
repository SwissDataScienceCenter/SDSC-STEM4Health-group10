package com.bioreactor.dtb;

import com.bioreactor.engine.*;

import java.util.*;

/**
 * DTB V3-GI model — V3 with Growth Inhibition by byproduct.
 *
 * This adds the K_IP parameter from the team's best Python model (V3_GI):
 *   mu = mu_max * S/(Ks+S) * N/(K_N+N) * K_IP/(K_IP+P)
 *
 * Python V3_GI achieved: VCD=13.4% TCD=13.2% Sub=19.4% Byp=18.5% Mean=16.2%
 * (those numbers are on all batches including training, not clean holdout)
 * The best holdout MAPE is VCD~8.5% TCD~8.5% from V3.
 *
 * 6 states + 2 hidden = 8: [Xv, Xd, S, P, B, DO, V, T, pH, N, D]
 * Hidden: N = nutrient (N0=1), D = damage (undiluted)
 */
public class DTBV3GIModel implements ProcessModel {

    private static final double EPS_S = 0.05;

    public static final int HIDDEN_N = 0;
    public static final int HIDDEN_D = 1;

    @Override
    public int hiddenStateCount() { return 2; }

    @Override
    public double[] defaultHiddenState() { return new double[]{1.0, 0.0}; }

    @Override
    public Set<Integer> undilutedHiddenIndices() { return Set.of(HIDDEN_D); }

    @Override
    public List<ModelParameter> parameters() {
        // Defaults from Python V3-GI full-fit (results/v3_gi/params_full.csv)
        // then adjusted to be within reasonable ranges for fitting
        return List.of(
            // Growth
            new ModelParameter("mu_max",   4.706, 0.5,  20.0,  "1/d",         "Max specific growth rate"),
            new ModelParameter("Ks",       2.707, 1e-3, 50.0,  "g/L",         "Monod half-saturation for substrate"),
            new ModelParameter("K_N",      0.353, 1e-3, 10.0,  "N0",          "Half-saturation on hidden nutrient"),
            new ModelParameter("Yxn",     41.966, 0.1,  1000.0,"cells/N0",    "Biomass yield on hidden nutrient"),
            new ModelParameter("C_Nb",     0.0001,1e-4, 1000.0,"N0",          "Hidden nutrient conc in Feed B"),

            // Death
            new ModelParameter("kd0",      0.000001, 1e-7, 0.5,   "1/d",      "Basal death rate"),
            new ModelParameter("kd_N",     0.0317, 1e-6, 2.0,   "1/d",        "Extra death when nutrient depleted (H1)"),
            new ModelParameter("K_dN",     0.00036, 1e-5, 10.0, "N0",         "Nutrient level for H1 death half-max"),
            new ModelParameter("kd_D",     0.00437, 1e-7, 1.0,  "1/(d·g/L·d)","Death per unit byproduct damage (H2)"),
            new ModelParameter("k_r",      1.091, 1e-4, 10.0,  "1/d",         "Decay/repair rate of byproduct damage"),
            new ModelParameter("k_lys",    0.0491, 1e-5, 5.0,  "1/d",         "Lysis rate of dead cells"),

            // Substrate uptake
            new ModelParameter("Yxs",     10000.0, 0.1,  1e4,  "cells/g",     "Biomass yield on substrate (growth)"),
            new ModelParameter("m",        0.0580, 1e-4, 2.0,  "g/(cells·d)","Maintenance substrate uptake"),

            // Overflow metabolism
            new ModelParameter("q_ovf",    1.932, 1e-3, 5.0,   "g/(cells·d)","Max overflow uptake"),
            new ModelParameter("K_ovf",    2.614, 0.3,  10.0,  "g/L",         "Half-saturation for overflow"),
            new ModelParameter("n_ovf",   20.0,   1.0,  20.0,  "-",           "Hill coefficient for overflow switch"),
            new ModelParameter("K_iP",     0.176, 1e-3, 50.0,  "g/L",         "Byproduct level halving overflow (lactate shift)"),

            // Byproduct
            new ModelParameter("Yps",      1.336, 0.05, 5.0,   "g/g",          "Byproduct yield from overflow"),
            new ModelParameter("qPc",      2.0,   1e-5, 2.0,   "g/(cells·d)","Max byproduct consumption"),
            new ModelParameter("K_Pc",    28.359, 1e-3, 100.0,"g/L",          "Half-sat for byproduct consumption"),

            // Growth inhibition by byproduct (NEW in V3-GI)
            new ModelParameter("K_IP",     0.758, 1e-2, 1000.0,"g/L",          "Byproduct level halving growth"),

            // Feed concentrations
            new ModelParameter("C_sub",    1.0,   1.0,  3000.0,"g/L",         "Substrate conc in substrate feed"),
            new ModelParameter("C_b",     74.456, 0.1,  1000.0,"g/L",         "Substrate conc in Feed B")
        );
    }

    @Override
    public double[] rates(BioreactorState state, Map<String, Double> p) {
        double mu_max = p.getOrDefault("mu_max", 4.706);
        double Ks     = p.getOrDefault("Ks", 2.707);
        double K_N    = p.getOrDefault("K_N", 0.353);
        double Yxn    = p.getOrDefault("Yxn", 41.966);
        double kd0    = p.getOrDefault("kd0", 0.000001);
        double kd_N   = p.getOrDefault("kd_N", 0.0317);
        double K_dN   = p.getOrDefault("K_dN", 0.00036);
        double kd_D   = p.getOrDefault("kd_D", 0.00437);
        double k_r    = p.getOrDefault("k_r", 1.091);
        double k_lys  = p.getOrDefault("k_lys", 0.0491);
        double Yxs    = p.getOrDefault("Yxs", 10000.0);
        double m      = p.getOrDefault("m", 0.0580);
        double q_ovf  = p.getOrDefault("q_ovf", 1.932);
        double K_ovf  = p.getOrDefault("K_ovf", 2.614);
        double n_ovf  = p.getOrDefault("n_ovf", 20.0);
        double K_iP   = p.getOrDefault("K_iP", 0.176);
        double Yps    = p.getOrDefault("Yps", 1.336);
        double qPc    = p.getOrDefault("qPc", 2.0);
        double K_Pc   = p.getOrDefault("K_Pc", 28.359);
        double K_IP   = p.getOrDefault("K_IP", 0.758);

        double Xv = Math.max(0, state.xv());
        double Xd = Math.max(0, state.xd());
        double S  = Math.max(0, state.substrate());
        double P  = Math.max(0, state.byproduct());
        double N  = Math.max(0, state.hidden(HIDDEN_N));
        double Dm = Math.max(0, state.hidden(HIDDEN_D));

        double fS = S / (EPS_S + S);

        // Growth: Monod × hidden nutrient × byproduct inhibition (NEW)
        double mu = mu_max * S / (Ks + S) * N / (K_N + N) * K_IP / (K_IP + P);

        // Death: basal + H1 (nutrient depletion) + H2 (damage)
        double kd = kd0 + kd_N * K_dN / (K_dN + N) + kd_D * Dm;

        // Overflow uptake with Hill function, shut off by byproduct (lactate shift)
        double Sn = Math.pow(S, n_ovf);
        double q_ovf_val = q_ovf * Sn / (Math.pow(K_ovf, n_ovf) + Sn) * K_iP / (K_iP + P);

        // Substrate uptake
        double qS = mu / Yxs + (m + q_ovf_val) * fS;

        // Byproduct: produced from overflow, consumed at low P
        double qP = Yps * q_ovf_val * fS - qPc * P / (K_Pc + P);

        // Nutrient consumption by growth
        double dN = -mu / Yxn * Xv;

        // Damage accumulation: proportional to byproduct, with repair
        double dD = P - k_r * Dm;

        double dXv = (mu - kd) * Xv;
        double dXd = kd * Xv - k_lys * Xd;
        double dS  = -qS * Xv;
        double dP  = qP * Xv;

        return new double[]{dXv, dXd, dS, 0.0, dP, 0.0, 0.0, 0.0, 0.0, dN, dD};
    }

    @Override
    public Map<String, Double> feedConcentrations(Map<String, Double> p) {
        double C_sub = p.getOrDefault("C_sub", 1.0);
        double C_b   = p.getOrDefault("C_b", 74.456);
        return Map.of("substrate", C_sub, "feed_b", C_b);
    }

    @Override
    public Map<String, Double> feedStreamConcentrations(String feedName, Map<String, Double> p) {
        return switch (feedName) {
            case "feed_b" -> {
                double C_b  = p.getOrDefault("C_b", 74.456);
                double C_Nb = p.getOrDefault("C_Nb", 0.0001);
                Map<String, Double> m = new HashMap<>();
                m.put("substrate", C_b);
                m.put("hidden_0", C_Nb);
                yield m;
            }
            case "substrate", "feed_substrate" -> Map.of("substrate",
                p.getOrDefault("C_sub", 1.0));
            default -> Map.of();
        };
    }

    /** Convenience: all params as Map with defaults. */
    public Map<String, Double> defaultParams() {
        Map<String, Double> p = new HashMap<>();
        for (ModelParameter mp : parameters()) p.put(mp.name(), mp.startValue());
        return p;
    }
}

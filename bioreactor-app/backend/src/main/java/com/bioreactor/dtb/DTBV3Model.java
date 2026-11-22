package com.bioreactor.dtb;

import com.bioreactor.engine.*;

import java.util.*;

/**
 * DTB V3 mechanistic model — port of the team's best Python model.
 *
 * Ported from: origin/main:src/mechanistic.py — rates_v3(), PARAMS_V3, V3 model.
 *
 * 8 states (6 core + 2 hidden):
 *   Core:  [Xv, Xd, S, P(primary, unused), B(byproduct), DO(unused), V(unused), T(unused), pH(unused)]
 *   Hidden: [N = limiting nutrient (normalised, N(0)=1), D = byproduct damage (D(0)=0)]
 *
 * Key mechanisms (from Python V3):
 *   1. Growth: μ = μ_max · S/(K_s+S) · N/(K_N+N)
 *   2. Lactate shift: overflow q_ovf shut off by byproduct: q_ovf *= K_iP/(K_iP+P)
 *   3. Death: kd = kd0 + kd_N·K_dN/(K_dN+N) + kd_D·D (H1 + H2)
 *   4. Damage: dD/dt = P - k_r·D (D undiluted by feeds)
 *   5. Overflow (Hill): q_ovf_max·Sⁿ/(K_ovfⁿ+Sⁿ)
 *   6. Byproduct: qP = Yps·q_ovf·fS - qPc·P/(K_Pc+P)
 *   7. Substrate: qS = μ/Yxs + (m + q_ovf)·fS
 */
public class DTBV3Model implements ProcessModel {

    private static final double EPS_S = 0.05;

    public static final int HIDDEN_N = 0;  // hidden[0] = nutrient N
    public static final int HIDDEN_D = 1;  // hidden[1] = damage D (undiluted)

    @Override
    public int hiddenStateCount() {
        return 2;
    }

    /** Indices of hidden states that are NOT diluted by feeds. D is culture history. */
    @Override
    public Set<Integer> undilutedHiddenIndices() {
        return Set.of(HIDDEN_D);
    }

    @Override
    public List<ModelParameter> parameters() {
        // Defaults from team's Python V3 full-fit (results/v3/params_full.csv).
        // These produced CV MAPE: VCD 8.6%, TCD 8.5%, Sub 19.8%, Byp 18.5%.
        // Ported and verified in Java: same parameters give MAPE 14.6% on holdout.
        return List.of(
            // Growth
            new ModelParameter("mu_max",  11.163, 0.1,  20.0,  "1/d",         "Max specific growth rate"),
            new ModelParameter("Ks",       1.178, 1e-3, 50.0,  "g/L",         "Monod half-saturation for substrate"),
            new ModelParameter("K_N",     10.0,   1e-3, 10.0,  "N0",          "Half-saturation on hidden nutrient"),
            new ModelParameter("Yxn",     45.638, 0.1,  1000.0,"cells/N0",    "Biomass yield on hidden nutrient"),
            new ModelParameter("C_Nb",     0.0001,1e-4, 1000.0,"N0",          "Hidden nutrient conc in Feed B"),

            // Death
            new ModelParameter("kd0",      0.000001, 1e-6, 0.5,   "1/d",      "Basal death rate"),
            new ModelParameter("kd_N",     2.0,   1e-6, 2.0,   "1/d",         "Extra death when nutrient depleted (H1)"),
            new ModelParameter("K_dN",     0.000277, 1e-4, 10.0, "N0",        "Nutrient level for H1 death half-max"),
            new ModelParameter("kd_D",     0.04581, 1e-7, 1.0,  "1/(d·g/L·d)","Death per unit byproduct damage (H2)"),
            new ModelParameter("k_r",      9.983, 1e-4, 10.0,  "1/d",         "Decay/repair rate of byproduct damage"),
            new ModelParameter("k_lys",    0.3083, 1e-5, 5.0,  "1/d",         "Lysis rate of dead cells"),

            // Substrate uptake
            new ModelParameter("Yxs",     10000.0, 0.1,  1e4,  "cells/g",    "Biomass yield on substrate (growth)"),
            new ModelParameter("m",        0.06193, 1e-4, 2.0,  "g/(cells·d)","Maintenance substrate uptake"),

            // Overflow metabolism
            new ModelParameter("q_ovf",    5.0,   1e-3, 5.0,   "g/(cells·d)","Max overflow uptake"),
            new ModelParameter("K_ovf",    2.706, 0.3,  10.0,  "g/L",         "Half-saturation for overflow"),
            new ModelParameter("n_ovf",   20.0,   1.0,  20.0,  "-",           "Hill coefficient for overflow switch"),
            new ModelParameter("K_iP",     0.08963, 1e-3, 50.0, "g/L",        "Byproduct level halving overflow (lactate shift)"),

            // Byproduct
            new ModelParameter("Yps",      1.0862, 0.05, 5.0,  "g/g",          "Byproduct yield from overflow"),
            new ModelParameter("qPc",      2.0,   1e-5, 2.0,   "g/(cells·d)","Max byproduct consumption"),
            new ModelParameter("K_Pc",    27.336, 1e-3, 100.0,"g/L",          "Half-sat for byproduct consumption"),

            // Feed concentrations
            new ModelParameter("C_sub",    1.0,   1.0,  3000.0,"g/L",         "Substrate conc in substrate feed"),
            new ModelParameter("C_b",     90.156, 0.1,  1000.0,"g/L",         "Substrate conc in Feed B")
        );
    }

    @Override
    public double[] rates(BioreactorState state, Map<String, Double> p) {
        double mu_max = p.getOrDefault("mu_max", 11.163);
        double Ks     = p.getOrDefault("Ks", 1.178);
        double K_N    = p.getOrDefault("K_N", 10.0);
        double Yxn    = p.getOrDefault("Yxn", 45.638);
        double kd0    = p.getOrDefault("kd0", 0.000001);
        double kd_N   = p.getOrDefault("kd_N", 2.0);
        double K_dN   = p.getOrDefault("K_dN", 0.000277);
        double kd_D   = p.getOrDefault("kd_D", 0.04581);
        double k_r    = p.getOrDefault("k_r", 9.983);
        double k_lys  = p.getOrDefault("k_lys", 0.3083);
        double Yxs    = p.getOrDefault("Yxs", 10000.0);
        double m      = p.getOrDefault("m", 0.06193);
        double q_ovf  = p.getOrDefault("q_ovf", 5.0);
        double K_ovf  = p.getOrDefault("K_ovf", 2.706);
        double n_ovf  = p.getOrDefault("n_ovf", 20.0);
        double K_iP   = p.getOrDefault("K_iP", 0.08963);
        double Yps    = p.getOrDefault("Yps", 1.0862);
        double qPc    = p.getOrDefault("qPc", 2.0);
        double K_Pc   = p.getOrDefault("K_Pc", 27.336);

        double Xv = Math.max(0, state.xv());
        double Xd = Math.max(0, state.xd());
        double S  = Math.max(0, state.substrate());
        double P  = Math.max(0, state.byproduct());
        double N  = Math.max(0, state.hidden(HIDDEN_N));
        double Dm = Math.max(0, state.hidden(HIDDEN_D));

        double fS = S / (EPS_S + S);

        // Growth: Monod × hidden nutrient
        double mu = mu_max * S / (Ks + S) * N / (K_N + N);

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

        // Core rates (9) + hidden rates (2) = 11
        return new double[]{dXv, dXd, dS, 0.0, dP, 0.0, 0.0, 0.0, 0.0, dN, dD};
    }

    @Override
    public Map<String, Double> feedConcentrations(Map<String, Double> p) {
        double C_sub = p.getOrDefault("C_sub", 1.0);
        double C_b   = p.getOrDefault("C_b", 90.156);
        return Map.of("substrate", C_sub, "feed_b", C_b);
    }

    /**
     * Feed B carries substrate AND hidden nutrient N.
     */
    @Override
    public Map<String, Double> feedStreamConcentrations(String feedName, Map<String, Double> p) {
        return switch (feedName) {
            case "feed_b" -> {
                double C_b  = p.getOrDefault("C_b", 90.156);
                double C_Nb = p.getOrDefault("C_Nb", 0.0001);
                Map<String, Double> m = new HashMap<>();
                m.put("substrate", C_b);
                m.put("hidden_0", C_Nb);  // nutrient N enters with Feed B
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

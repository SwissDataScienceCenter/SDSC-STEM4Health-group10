package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import java.util.*;

/**
 * DTB V2 model — logistic growth brake + overflow metabolism.
 *
 * Ported from team's mechanistic.py v2_logistic model.
 * State vector: [Xv, Xd, S, P, (hidden states...)]
 * This v2 uses 4 states (no hidden states).
 *
 * Features:
 *   - Logistic brake: mu *= (1 - Xv/Xmax)
 *   - Death: kd = kd0 + kd_S * K_dS/(K_dS + S)
 *   - Lysis: dXd -= k_lys * Xd
 *   - Overflow uptake: additional substrate consumption at high S
 *   - Byproduct from overflow, consumption at low P
 */
public class DTBV2Model implements ProcessModel {

    private static final double EPS_S = 0.05;
    private static final double EPS_P = 0.01;

    @Override
    public List<ModelParameter> parameters() {
        return List.of(
            // Growth
            new ModelParameter("mu_max",   1.0,   0.1,  5.0,   "1/d",      "Max specific growth rate"),
            new ModelParameter("Ks",       0.1,   1e-3, 5.0,   "g/L",      "Monod half-saturation for growth"),
            // Logistic brake
            new ModelParameter("Xmax",     32.0,  5.0,  200.0, "1e6/mL",   "Carrying capacity"),
            // Death/lysis
            new ModelParameter("kd0",      0.005, 1e-5, 0.5,   "1/d",      "Basal death rate"),
            new ModelParameter("kd_S",     0.05,  1e-5, 2.0,   "1/d",      "Extra death under low S"),
            new ModelParameter("K_dS",     0.5,   1e-2, 10.0,  "g/L",      "Half-sat for extra death"),
            new ModelParameter("k_lys",    0.1,   1e-4, 5.0,   "1/d",      "Lysis rate of dead cells"),
            // Substrate uptake
            new ModelParameter("Yxs",      5.0,   0.1,  200.0, "cells/g",  "Biomass yield (growth)"),
            new ModelParameter("m",        0.1,   1e-4, 2.0,   "g/(cells d)","Maintenance"),
            new ModelParameter("q_ovf",    0.6,   1e-3, 5.0,   "g/(cells d)","Max overflow uptake"),
            new ModelParameter("K_ovf",    3.0,   0.3,  10.0,  "g/L",      "Half-sat for overflow"),
            new ModelParameter("n_ovf",    4.0,   1.0,  12.0,  "-",        "Hill coeff for overflow"),
            // Byproduct
            new ModelParameter("Yps",      1.0,   0.05, 5.0,   "g/g",      "Byproduct yield from overflow"),
            new ModelParameter("qPc",      0.02,  1e-5, 2.0,   "g/(cells d)","Max byproduct consumption"),
            new ModelParameter("K_Pc",     0.5,   1e-3, 10.0,  "g/L",      "Half-sat for byproduct consumption"),
            // Feed concentrations
            new ModelParameter("C_sub",    250.0, 10.0, 3000.0,"g/L",      "Substrate in substrate feed"),
            new ModelParameter("C_b",      50.0,  0.1,  1000.0,"g/L",      "Substrate in Feed B")
        );
    }

    @Override
    public double[] rates(BioreactorState state, Map<String, Double> p) {
        double mu_max = p.getOrDefault("mu_max", 1.0);
        double Ks     = p.getOrDefault("Ks", 0.1);
        double Xmax   = p.getOrDefault("Xmax", 32.0);
        double kd0    = p.getOrDefault("kd0", 0.005);
        double kd_S   = p.getOrDefault("kd_S", 0.05);
        double K_dS   = p.getOrDefault("K_dS", 0.5);
        double k_lys  = p.getOrDefault("k_lys", 0.1);
        double Yxs    = p.getOrDefault("Yxs", 5.0);
        double m      = p.getOrDefault("m", 0.1);
        double q_ovf  = p.getOrDefault("q_ovf", 0.6);
        double K_ovf  = p.getOrDefault("K_ovf", 3.0);
        double n_ovf  = p.getOrDefault("n_ovf", 4.0);
        double Yps    = p.getOrDefault("Yps", 1.0);
        double qPc    = p.getOrDefault("qPc", 0.02);
        double K_Pc   = p.getOrDefault("K_Pc", 0.5);

        double Xv = Math.max(0, state.xv());
        double Xd = Math.max(0, state.xd());
        double S  = Math.max(0, state.substrate());
        double P  = Math.max(0, state.byproduct()); // byproduct is stored in index 4

        double fS = S / (EPS_S + S);

        // Growth with logistic brake
        double brake = Math.max(0, 1.0 - Xv / Xmax);
        double mu = mu_max * S / (Ks + S) * brake;

        // Death + lysis
        double kd = kd0 + kd_S * K_dS / (K_dS + S);

        // Overflow uptake (Hill function)
        double Sn = Math.pow(S, n_ovf);
        double q_ovf_val = q_ovf * Sn / (Math.pow(K_ovf, n_ovf) + Sn);

        // Total substrate uptake
        double qS = mu / Yxs + (m + q_ovf_val) * fS;

        // Byproduct: produced from overflow, consumed at low P
        double qP = Yps * q_ovf_val * fS - qPc * P / (K_Pc + P);

        double dXv = (mu - kd) * Xv;
        double dXd = kd * Xv - k_lys * Xd;
        double dS  = -qS * Xv;
        double dP  = qP * Xv;

        return new double[]{dXv, dXd, dS, 0.0, dP, 0.0, 0.0, 0.0, 0.0};
    }

    @Override
    public Map<String, Double> feedConcentrations(Map<String, Double> p) {
        double C_sub = p.getOrDefault("C_sub", 250.0);
        double C_b   = p.getOrDefault("C_b", 50.0);
        return Map.of("substrate", C_sub, "feed_b", C_b);
    }

    @Override
    public Map<String, Double> feedStreamConcentrations(String feedName, Map<String, Double> p) {
        return switch (feedName) {
            case "feed_b", "substrate", "feed_substrate" -> {
                double c = p.getOrDefault(feedName.equals("feed_b") ? "C_b" : "C_sub", 250.0);
                yield Map.of("substrate", c);
            }
            case "feed_c", "feed_d" -> Map.of(); // inert feeds, no substrate
            default -> Map.of();
        };
    }

    /** Convenience: params as a Map with defaults. */
    public Map<String, Double> defaultParams() {
        Map<String, Double> p = new HashMap<>();
        for (ModelParameter mp : parameters()) p.put(mp.name(), mp.startValue());
        return p;
    }
}

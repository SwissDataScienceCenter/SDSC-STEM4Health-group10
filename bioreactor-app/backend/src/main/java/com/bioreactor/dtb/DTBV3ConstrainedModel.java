package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import java.util.*;

/** DTB V3 CONSTRAINED — 12 free params, 10 locked to CHO literature values. */
public class DTBV3ConstrainedModel implements ProcessModel {

    private static final double EPS_S = 0.05;
    public static final int HIDDEN_N = 0;
    public static final int HIDDEN_D = 1;

    private static final double K_N  = 1.78, Yxn = 43.6, C_Nb = 0.03;
    private static final double kd0 = 0.005, kd_N = 0.10, K_dN = 0.10;
    private static final double kd_D = 0.05, k_r = 1.0, k_lys = 0.02, K_Pc = 5.0;

    @Override public int hiddenStateCount() { return 2; }
    @Override public Set<Integer> undilutedHiddenIndices() { return Set.of(HIDDEN_D); }

    @Override
    public List<ModelParameter> parameters() {
        return List.of(
            new ModelParameter("mu_max",   1.0,   0.5,  2.0,   "1/d",         "Max growth rate [CHO]"),
            new ModelParameter("Ks",       4.9,   0.1,  50.0,  "g/L",         "Substrate half-sat"),
            new ModelParameter("Yxs",      200.0, 50.0, 500.0, "cells/g",     "Biomass yield on sub"),
            new ModelParameter("m",        0.1,   0.001,0.5,   "g/(cells·d)","Maintenance"),
            new ModelParameter("q_ovf",    0.47,  0.01, 5.0,   "g/(cells·d)","Max overflow uptake"),
            new ModelParameter("K_ovf",    3.6,   0.1,  10.0,  "g/L",         "Overflow half-sat"),
            new ModelParameter("n_ovf",    8.0,   1.0,  20.0,  "-",           "Overflow Hill n"),
            new ModelParameter("K_iP",     0.5,   0.01, 10.0,  "g/L",         "Byp level halving ovf"),
            new ModelParameter("Yps",      1.0,   0.1,  3.0,   "g/g",         "Byp yield from overflow"),
            new ModelParameter("qPc",      0.1,   0.001,2.0,   "g/(cells·d)","Max byp consumption"),
            new ModelParameter("C_sub",    250.0, 50.0, 1000.0,"g/L",         "Substrate feed conc"),
            new ModelParameter("C_b",      50.0,  5.0,  500.0, "g/L",         "Feed B substrate conc")
        );
    }

    @Override
    public double[] rates(BioreactorState s, Map<String, Double> p) {
        double mu_max = p.getOrDefault("mu_max", 1.0);
        double Ks     = p.getOrDefault("Ks", 4.9);
        double Yxs    = p.getOrDefault("Yxs", 200.0);
        double m      = p.getOrDefault("m", 0.1);
        double q_ovf  = p.getOrDefault("q_ovf", 0.47);
        double K_ovf  = p.getOrDefault("K_ovf", 3.6);
        double n_ovf  = p.getOrDefault("n_ovf", 8.0);
        double K_iP   = p.getOrDefault("K_iP", 0.5);
        double Yps    = p.getOrDefault("Yps", 1.0);
        double qPc    = p.getOrDefault("qPc", 0.1);

        double Xv = Math.max(0, s.xv()), Xd = Math.max(0, s.xd());
        double S = Math.max(0, s.substrate()), P = Math.max(0, s.byproduct());
        double N = Math.max(0, s.hidden(HIDDEN_N)), Dm = Math.max(0, s.hidden(HIDDEN_D));
        double fS = S / (EPS_S + S);

        double mu = mu_max * S / (Ks + S) * N / (K_N + N);
        double kd = kd0 + kd_N * K_dN / (K_dN + N) + kd_D * Dm;
        double Sn = Math.pow(S, n_ovf);
        double q_ovf_val = q_ovf * Sn / (Math.pow(K_ovf, n_ovf) + Sn) * K_iP / (K_iP + P);
        double qS = mu / Yxs + (m + q_ovf_val) * fS;
        double qP = Yps * q_ovf_val * fS - qPc * P / (K_Pc + P);

        return new double[]{
            (mu - kd) * Xv, kd * Xv - k_lys * Xd, -qS * Xv, 0.0, qP * Xv,
            0.0, 0.0, 0.0, 0.0, -mu / Yxn * Xv, P - k_r * Dm
        };
    }

    @Override public Map<String, Double> feedConcentrations(Map<String, Double> p) {
        return Map.of("substrate", p.getOrDefault("C_sub", 250.0),
                      "feed_b", p.getOrDefault("C_b", 50.0));
    }

    @Override
    public Map<String, Double> feedStreamConcentrations(String feedName, Map<String, Double> p) {
        return switch (feedName) {
            case "feed_b" -> {
                Map<String, Double> m = new HashMap<>();
                m.put("substrate", p.getOrDefault("C_b", 50.0));
                m.put("hidden_0", C_Nb);
                yield m;
            }
            case "substrate", "feed_substrate" -> Map.of("substrate",
                p.getOrDefault("C_sub", 250.0));
            default -> Map.of();
        };
    }
}

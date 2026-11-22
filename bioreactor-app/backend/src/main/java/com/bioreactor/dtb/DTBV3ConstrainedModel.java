package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import java.util.*;

/** DTB V3 CONSTRAINED — 12 free params, 10 locked to CHO literature values. */
public class DTBV3ConstrainedModel implements ProcessModel {

    private static final double EPS_S = 0.05;
    public static final int HIDDEN_N = 0;
    public static final int HIDDEN_D = 1;

    // Locked parameters from team Python V3 fit (well-identified, at bounds or near-zero)
    private static final double K_N  = 10.0,  Yxn = 45.638, C_Nb = 0.0001;
    private static final double kd0 = 0.000001, kd_N = 2.0, K_dN = 0.000277;
    private static final double kd_D = 0.04581, k_r = 9.983, k_lys = 0.3083, K_Pc = 27.336;

    @Override public int hiddenStateCount() { return 2; }
    @Override public Set<Integer> undilutedHiddenIndices() { return Set.of(HIDDEN_D); }

    @Override
    public List<ModelParameter> parameters() {
        // 12 free parameters, starting from Python V3 fit values
        return List.of(
            new ModelParameter("mu_max",  11.163, 0.5,  20.0,  "1/d",         "Max growth rate"),
            new ModelParameter("Ks",       1.178, 0.1,  50.0,  "g/L",         "Substrate half-sat"),
            new ModelParameter("Yxs",     10000.0,50.0, 10000.0,"cells/g",    "Biomass yield on sub"),
            new ModelParameter("m",        0.062, 0.001,2.0,   "g/(cells·d)","Maintenance"),
            new ModelParameter("q_ovf",    5.0,   0.01, 5.0,   "g/(cells·d)","Max overflow uptake"),
            new ModelParameter("K_ovf",    2.706, 0.1,  10.0,  "g/L",         "Overflow half-sat"),
            new ModelParameter("n_ovf",   20.0,   1.0,  20.0,  "-",           "Overflow Hill n"),
            new ModelParameter("K_iP",     0.0896,0.01, 10.0,  "g/L",         "Byp level halving ovf"),
            new ModelParameter("Yps",      1.086, 0.1,  3.0,   "g/g",         "Byp yield from overflow"),
            new ModelParameter("qPc",      2.0,   0.001,2.0,   "g/(cells·d)","Max byp consumption"),
            new ModelParameter("C_sub",    1.0,   0.5,  3000.0,"g/L",         "Substrate feed conc"),
            new ModelParameter("C_b",     90.156, 5.0,  500.0, "g/L",         "Feed B substrate conc")
        );
    }

    @Override
    public double[] rates(BioreactorState s, Map<String, Double> p) {
        double mu_max = p.getOrDefault("mu_max", 11.163);
        double Ks     = p.getOrDefault("Ks", 1.178);
        double Yxs    = p.getOrDefault("Yxs", 10000.0);
        double m      = p.getOrDefault("m", 0.062);
        double q_ovf  = p.getOrDefault("q_ovf", 5.0);
        double K_ovf  = p.getOrDefault("K_ovf", 2.706);
        double n_ovf  = p.getOrDefault("n_ovf", 20.0);
        double K_iP   = p.getOrDefault("K_iP", 0.0896);
        double Yps    = p.getOrDefault("Yps", 1.086);
        double qPc    = p.getOrDefault("qPc", 2.0);

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
        return Map.of("substrate", p.getOrDefault("C_sub", 1.0),
                      "feed_b", p.getOrDefault("C_b", 90.156));
    }

    @Override
    public Map<String, Double> feedStreamConcentrations(String feedName, Map<String, Double> p) {
        return switch (feedName) {
            case "feed_b" -> {
                Map<String, Double> m = new HashMap<>();
                m.put("substrate", p.getOrDefault("C_b", 90.156));
                m.put("hidden_0", C_Nb);
                yield m;
            }
            case "substrate", "feed_substrate" -> Map.of("substrate",
                p.getOrDefault("C_sub", 1.0));
            default -> Map.of();
        };
    }
}

package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import java.util.*;

/**
 * DTB-specific process model — port of team's V1 mechanisms (mechanistic.py).
 *
 * Key differences from textbook MonodModel:
 *   - Time is in DAYS (CHO mammalian culture, ~14 d)
 *   - Cell units: 1e6 cells/mL (per spec: "VCD" = viable cell density)
 *     Y_XS is therefore "cells/g" not "g/g"
 *   - Smooth substrate switch-off (EPS_S = 0.05 g/L) to avoid negative S
 *   - Death rate constant (simple model)
 *   - Feed B and feed_substrate carry substrate; feed C and D are inert
 *   - Volume is reset to measured value at each sampling time
 */
public class DTBModel implements ProcessModel {

    private static final double EPS_S = 0.05;  // smooth switch-off for substrate uptake

    @Override
    public List<ModelParameter> parameters() {
        return List.of(
            // Growth
            new ModelParameter("mu_max",   1.0,   0.1,  5.0,   "1/d",      "Max specific growth rate"),
            new ModelParameter("Ks",       0.1,   1e-3, 5.0,   "g/L",      "Monod half-saturation"),
            new ModelParameter("kd",       0.02,  1e-5, 1.0,   "1/d",      "Specific death rate"),
            new ModelParameter("Yxs",      5.0,   0.1,  200.0, "cells/g",  "Biomass yield on substrate"),
            new ModelParameter("m",        0.1,   1e-5, 2.0,   "g/(cells d)","Maintenance coefficient"),
            // Byproduct (Luedeking-Piret)
            new ModelParameter("alpha",    0.6,   1e-5, 10.0,  "g/cells",  "Growth-associated byproduct"),
            new ModelParameter("beta",     0.01,  1e-6, 2.0,   "g/(cells d)","Non-growth byproduct rate"),
            // Feed concentrations
            new ModelParameter("C_sub",    250.0, 10.0, 3000.0,"g/L",      "Substrate conc in substrate feed"),
            new ModelParameter("C_b",      50.0,  0.1,  1000.0,"g/L",      "Substrate conc in Feed B")
        );
    }

    @Override
    public double[] rates(BioreactorState state, Map<String, Double> p) {
        double mu_max = p.getOrDefault("mu_max", 1.0);
        double Ks     = p.getOrDefault("Ks", 0.1);
        double kd     = p.getOrDefault("kd", 0.02);
        double Yxs    = p.getOrDefault("Yxs", 5.0);
        double m      = p.getOrDefault("m", 0.1);
        double alpha  = p.getOrDefault("alpha", 0.6);
        double beta   = p.getOrDefault("beta", 0.01);

        double Xv = Math.max(0, state.xv());
        double Xd = Math.max(0, state.xd());
        double S  = Math.max(0, state.substrate());

        // Smooth switch-off: fS = S / (EPS_S + S)
        double fS = S / (EPS_S + S);

        // Monod growth
        double mu = mu_max * S / (Ks + S);

        // Pirt uptake (with switch-off)
        double qS = mu / Yxs + m * fS;

        // Luedeking-Piret byproduct
        double qP = alpha * mu + beta;

        double dXv = (mu - kd) * Xv;
        double dXd = kd * Xv;
        double dS  = -qS * Xv;
        double dP  = qP * Xv;   // byproduct — index 4 in rates array is "byproduct"

        // returns [dXv, dXd, dS, dProduct(primary), dByproduct, dDO, dV, dT, dpH]
        return new double[]{dXv, dXd, dS, 0.0, dP, 0.0, 0.0, 0.0, 0.0};
    }

    @Override
    public Map<String, Double> feedConcentrations(Map<String, Double> p) {
        // Returns the concentration of each state in the feed.
        // State vector: [Xv, Xd, S, P(primary), B(byproduct), DO, V, T, pH]
        // Only substrate (index 2) enters with the feed.
        double C_sub = p.getOrDefault("C_sub", 250.0);
        double C_b   = p.getOrDefault("C_b", 50.0);

        // Default feed: substrate feed and Feed B carry substrate; Feed C/D carry 0
        return Map.of(
            "substrate", C_sub,
            "feed_b", C_b
        );
    }

    /**
     * Feed concentration vector for a specific feed stream.
     * Returns the full 9-state concentration vector for mass balance calculations.
     * Index: 0=Xv, 1=Xd, 2=S(glucose), 3=P(primary), 4=B(byproduct), 5=DO, 6=V, 7=T, 8=pH
     */
    public static double[] feedConcentrationVector(String feedName, Map<String, Double> params) {
        double[] c = new double[9];
        switch (feedName) {
            case "substrate", "feed_substrate" ->
                c[2] = params.getOrDefault("C_sub", 250.0);
            case "feed_b" ->
                c[2] = params.getOrDefault("C_b", 50.0);
            // feed_c and feed_d: no substrate (inert feeds)
        }
        return c;
    }
}

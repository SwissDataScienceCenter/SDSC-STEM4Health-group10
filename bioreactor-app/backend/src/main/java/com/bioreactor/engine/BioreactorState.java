package com.bioreactor.engine;

import java.util.Map;

/**
 * The state of the bioreactor at a point in time, with optional hidden states.
 *
 * Core fields (always present):
 *   xv (viable biomass), xd (dead biomass), substrate, product, byproduct,
 *   dissolvedO2, volume, temperature, pH
 *
 * Hidden states are appended after the core 8 fields in the state vector.
 * Models declare their hidden state count via ProcessModel.hiddenStateCount().
 */
public record BioreactorState(
    double time,
    double xv, double xd,
    double substrate, double product, double byproduct,
    double dissolvedO2, double volume,
    double temperature, double pH,
    double[] hidden  // model-specific hidden states (may be empty)
) {
    public static final int CORE_STATE_SIZE = 9; // xv..pH (without time)

    /** Construct with no hidden states (backward compatible). */
    public BioreactorState(double time, double xv, double xd,
                           double substrate, double product, double byproduct,
                           double dissolvedO2, double volume,
                           double temperature, double pH) {
        this(time, xv, xd, substrate, product, byproduct,
             dissolvedO2, volume, temperature, pH, new double[0]);
    }

    /** Get a hidden state by index (0-based). */
    public double hidden(int idx) {
        return idx < hidden.length ? hidden[idx] : 0;
    }

    /** Replace hidden states. */
    public BioreactorState withHidden(double[] newHidden) {
        return new BioreactorState(time, xv, xd, substrate, product, byproduct,
                                   dissolvedO2, volume, temperature, pH, newHidden);
    }

    /** Total state size including hidden states (for ODE). */
    public int stateSize() {
        return CORE_STATE_SIZE + hidden.length;
    }

    /** Full state vector for ODE integration: [xv, xd, S, P, B, DO, V, T, pH, h0, h1, ...] */
    public double[] stateVector() {
        double[] y = new double[stateSize()];
        y[0] = xv; y[1] = xd; y[2] = substrate;
        y[3] = product; y[4] = byproduct; y[5] = dissolvedO2;
        y[6] = volume; y[7] = temperature; y[8] = pH;
        System.arraycopy(hidden, 0, y, CORE_STATE_SIZE, hidden.length);
        return y;
    }

    /** Reconstruct from ODE state vector and time. */
    public static BioreactorState fromStateVector(double t, double[] y, int hiddenCount) {
        double xv = y[0], xd = y[1], s = y[2], p = y[3], b = y[4];
        double do2 = y[5], v = y[6], temp = y[7], ph = y[8];
        double[] h = new double[hiddenCount];
        if (hiddenCount > 0) System.arraycopy(y, CORE_STATE_SIZE, h, 0, hiddenCount);
        return new BioreactorState(t, xv, xd, s, p, b, do2, v, temp, ph, h);
    }

    /** Backward-compatible: assume 0 hidden states. */
    public static BioreactorState fromStateVector(double t, double[] y) {
        return fromStateVector(t, y, 0);
    }

    /** Apply a bolus addition: volume increases, concentrations follow mass balance.
     *  Hidden states are diluted unless listed in undilutedIndices. */
    public BioreactorState applyBolus(double dV, Map<String, Double> feedConcentrations) {
        return applyBolus(dV, feedConcentrations, new int[0]);
    }

    public BioreactorState applyBolus(double dV, Map<String, Double> feedConcentrations,
                                       int[] undilutedIndices) {
        double Vnew = this.volume + dV;
        double ratio = this.volume / Vnew;
        double feedRatio = dV / Vnew;
        double[] newHidden = new double[hidden.length];
        for (int i = 0; i < hidden.length; i++) {
            boolean undiluted = false;
            for (int ui : undilutedIndices) if (ui == i) { undiluted = true; break; }
            newHidden[i] = undiluted ? hidden[i] : hidden[i] * ratio;
        }
        // Feed concentrations can carry hidden state values (key = "hidden_0", "hidden_1", ...)
        for (int i = 0; i < hidden.length; i++) {
            String key = "hidden_" + i;
            if (feedConcentrations.containsKey(key)) {
                newHidden[i] = newHidden[i] + feedConcentrations.get(key) * feedRatio;
            }
        }
        return new BioreactorState(time,
            xv * ratio, xd * ratio,
            substrate * ratio + feedConcentrations.getOrDefault("substrate", 0.0) * feedRatio,
            product * ratio + feedConcentrations.getOrDefault("product", 0.0) * feedRatio,
            byproduct * ratio + feedConcentrations.getOrDefault("byproduct", 0.0) * feedRatio,
            dissolvedO2 * ratio, Vnew, temperature, pH, newHidden);
    }

    /** Set volume to a fixed value (used at sampling times for DTB). */
    public BioreactorState withVolume(double V) {
        return new BioreactorState(time, xv, xd, substrate, product, byproduct,
                                   dissolvedO2, V, temperature, pH, hidden);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("t=%.4f: Xv=%.4f, Xd=%.4f, S=%.4f, B=%.4f, V=%.4f",
            time, xv, xd, substrate, byproduct, volume));
        for (int i = 0; i < hidden.length; i++) {
            sb.append(String.format(", h%d=%.4f", i, hidden[i]));
        }
        return sb.toString();
    }
}

package com.bioreactor.engine;

import java.util.Map;

/**
 * The state of the bioreactor at a point in time.
 */
public record BioreactorState(
    double time,         // elapsed time, same unit as model
    double xv,           // viable biomass concentration
    double xd,           // dead biomass concentration
    double substrate,    // substrate concentration
    double product,      // product concentration
    double byproduct,    // byproduct concentration
    double dissolvedO2,  // dissolved oxygen
    double volume,       // working volume
    double temperature,  // temperature
    double pH            // pH
) {
    public static final int STATE_SIZE = 10;

    public double[] toArray() {
        return new double[]{time, xv, xd, substrate, product, byproduct,
                            dissolvedO2, volume, temperature, pH};
    }

    public static BioreactorState fromArray(double t, double[] y) {
        return new BioreactorState(t, y[0], y[1], y[2], y[3], y[4],
                                   y[5], y[6], y[7], y[8]);
    }

    /** State portion for ODE integration (time is handled by the integrator). */
    public double[] stateVector() {
        return new double[]{xv, xd, substrate, product, byproduct,
                            dissolvedO2, volume, temperature, pH};
    }

    public static BioreactorState fromStateVector(double t, double[] y) {
        return new BioreactorState(t, y[0], y[1], y[2], y[3], y[4],
                                   y[5], y[6], y[7], y[8]);
    }

    /** Apply a bolus addition: volume increases, concentrations follow mass balance. */
    public BioreactorState applyBolus(double dV, Map<String, Double> feedConcentrations) {
        double Vnew = this.volume + dV;
        double ratio = this.volume / Vnew;
        double feedRatio = dV / Vnew;
        return new BioreactorState(
            time,
            xv * ratio,
            xd * ratio,
            substrate * ratio + feedConcentrations.getOrDefault("substrate", 0.0) * feedRatio,
            product * ratio + feedConcentrations.getOrDefault("product", 0.0) * feedRatio,
            byproduct * ratio + feedConcentrations.getOrDefault("byproduct", 0.0) * feedRatio,
            dissolvedO2 * ratio,
            Vnew,
            temperature,
            pH
        );
    }

    /** Set volume to a fixed value (used at sampling times for DTB). */
    public BioreactorState withVolume(double V) {
        return new BioreactorState(time, xv, xd, substrate, product, byproduct,
                                   dissolvedO2, V, temperature, pH);
    }
}

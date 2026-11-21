package com.bioreactor.engine;

/**
 * Temperature and pH modifiers for growth rates.
 * Ported from Python: simulation_engine.temp_factor_cardinal, ph_factor_gaussian.
 */
public final class EnvironmentFactors {

    private EnvironmentFactors() {}

    /** Cardinal temperature model: returns 0..1. */
    public static double tempFactorCardinal(double T, double Tmin, double Topt, double Tmax) {
        if (T <= Tmin || T >= Tmax) return 0.0;
        if (T <= Topt) {
            double d = (T - Tmin) / (Topt - Tmin);
            return d * d;
        } else {
            double d = (Tmax - T) / (Tmax - Topt);
            return d * d;
        }
    }

    /** Gaussian pH factor. */
    public static double phFactorGaussian(double pH, double pHopt, double sigma) {
        double d = (pH - pHopt) / sigma;
        return Math.exp(-d * d);
    }
}

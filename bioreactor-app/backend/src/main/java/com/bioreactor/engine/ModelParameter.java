package com.bioreactor.engine;

/**
 * A parameter in a process model, with bounds for fitting.
 * Ported from Python: simulation_engine.py dataclass fields.
 */
public record ModelParameter(
    String name,
    double startValue,
    double lowerBound,
    double upperBound,
    String unit,
    String description
) {
    /** Maps the parameter to the unconstrained domain (log for strictly positive). */
    public double toUnconstrained(double value) {
        if (lowerBound == 0 && upperBound == Double.POSITIVE_INFINITY) {
            return Math.log(Math.max(value, 1e-12));
        }
        if (lowerBound == Double.NEGATIVE_INFINITY && upperBound == Double.POSITIVE_INFINITY) {
            return value;
        }
        return Math.log(Math.max(value - lowerBound, 1e-12) / Math.max(upperBound - value, 1e-12));
    }

    /** Inverse of toUnconstrained. */
    public double fromUnconstrained(double u) {
        if (lowerBound == 0 && upperBound == Double.POSITIVE_INFINITY) {
            return Math.exp(u);
        }
        if (lowerBound == Double.NEGATIVE_INFINITY && upperBound == Double.POSITIVE_INFINITY) {
            return u;
        }
        double expU = Math.exp(u);
        return (lowerBound + upperBound * expU) / (1.0 + expU);
    }
}

package com.bioreactor.monitoring;

import java.util.*;

/**
 * Monitoring and early warning engine.
 *
 * Checks measured values against reference bands and predicted values against limits,
 * generating findings that must be acknowledged or dismissed by a person.
 */
public class MonitoringEngine {

    /**
     * Reference band: percentiles of training runs on a common time grid.
     */
    public record ReferenceBand(
        String parameterCode,
        double[] timeGrid,
        double[] p10,
        double[] p50,
        double[] p90
    ) {
        /**
         * Check if a value is outside the band.
         * Returns null if within band, or a description of the deviation.
         */
        public String check(double time, double value) {
            double p10Val = valueAt(time, p10);
            double p90Val = valueAt(time, p90);
            if (value < p10Val) {
                return String.format("%s=%.4f below 10th percentile %.4f at t=%.2f",
                    parameterCode, value, p10Val, time);
            }
            if (value > p90Val) {
                return String.format("%s=%.4f above 90th percentile %.4f at t=%.2f",
                    parameterCode, value, p90Val, time);
            }
            return null;
        }

        private double valueAt(double t, double[] grid) {
            if (t <= timeGrid[0]) return grid[0];
            if (t >= timeGrid[timeGrid.length - 1]) return grid[timeGrid.length - 1];
            for (int i = 0; i < timeGrid.length - 1; i++) {
                if (t >= timeGrid[i] && t < timeGrid[i + 1]) {
                    double frac = (t - timeGrid[i]) / (timeGrid[i + 1] - timeGrid[i]);
                    return grid[i] + frac * (grid[i + 1] - grid[i]);
                }
            }
            return grid[grid.length - 1];
        }
    }

    /**
     * Build reference bands from a list of training runs.
     * Each training state is: times[i], values[i] for a given parameter.
     */
    public static ReferenceBand buildBand(
        String parameterCode,
        List<double[]> timeSeries,   // each element: array of times for one run
        List<double[]> valueSeries,  // each element: corresponding values
        double[] commonGrid
    ) {
        int nRuns = timeSeries.size();
        int nGrid = commonGrid.length;

        double[][] interpolated = new double[nRuns][nGrid];
        for (int r = 0; r < nRuns; r++) {
            double[] tVals = timeSeries.get(r);
            double[] vVals = valueSeries.get(r);
            for (int g = 0; g < nGrid; g++) {
                interpolated[r][g] = interpolate(tVals, vVals, commonGrid[g]);
            }
        }

        double[] p10 = new double[nGrid];
        double[] p50 = new double[nGrid];
        double[] p90 = new double[nGrid];

        for (int g = 0; g < nGrid; g++) {
            double[] col = new double[nRuns];
            for (int r = 0; r < nRuns; r++) col[r] = interpolated[r][g];
            Arrays.sort(col);
            p10[g] = col[(int)(nRuns * 0.10)];
            p50[g] = col[(int)(nRuns * 0.50)];
            p90[g] = col[(int)(nRuns * 0.90)];
        }

        return new ReferenceBand(parameterCode, commonGrid, p10, p50, p90);
    }

    private static double interpolate(double[] times, double[] values, double t) {
        if (t <= times[0]) return values[0];
        if (t >= times[times.length - 1]) return values[times.length - 1];
        for (int i = 0; i < times.length - 1; i++) {
            if (t >= times[i] && t < times[i + 1]) {
                double frac = (t - times[i]) / (times[i + 1] - times[i]);
                return values[i] + frac * (values[i + 1] - values[i]);
            }
        }
        return values[values.length - 1];
    }

    public enum FindingKind {
        OUTSIDE_REFERENCE_BAND,
        PREDICTED_LIMIT_BREACH,
        MODEL_MISMATCH
    }

    public enum Severity { INFO, WARNING, ALARM }

    public record Finding(
        String runId, String parameterCode, double elapsedTime,
        FindingKind kind, Severity severity, String message
    ) {}
}

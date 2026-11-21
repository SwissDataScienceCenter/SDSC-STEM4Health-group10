package com.bioreactor.trending;

import java.util.*;

/**
 * Trending engine: control charts and capability indices across batches.
 *
 * Two rules to start:
 * 1. One point beyond 3 sigma (Shewhart rule for individual values).
 * 2. 8 points in a row on one side of the centre line (Nelson rule).
 */
public class TrendingEngine {

    public record ControlChart(
        String metricCode,
        double[] values,        // individual values per batch, in time order
        double[] movingRanges,  // |x_i - x_{i-1}|
        double centreLine,
        double ucl,             // upper control limit (centreLine + 3*sigma)
        double lcl,             // lower control limit (centreLine - 3*sigma)
        List<String> violations
    ) {}

    /**
     * Individual values and moving range chart (I-MR).
     */
    public static ControlChart buildIMRChart(
        String metricCode, String[] batchIds, double[] values
    ) {
        int n = values.length;
        double mean = Arrays.stream(values).average().orElse(0);

        // Moving range: |x_i - x_{i-1}|
        double[] mr = new double[n];
        for (int i = 1; i < n; i++) mr[i] = Math.abs(values[i] - values[i - 1]);
        double mrBar = Arrays.stream(mr).average().orElse(0);

        // Sigma estimate from moving range: σ = MR_bar / d2(2) = MR_bar / 1.128
        double sigma = mrBar / 1.128;
        double ucl = mean + 3 * sigma;
        double lcl = mean - 3 * sigma;

        List<String> violations = new ArrayList<>();

        // Rule 1: one point beyond 3 sigma
        for (int i = 0; i < n; i++) {
            if (values[i] > ucl) {
                violations.add(String.format("Batch %s (idx %d): value %.4f > UCL %.4f (Rule 1)",
                    batchIds[i], i, values[i], ucl));
            }
            if (values[i] < lcl) {
                violations.add(String.format("Batch %s (idx %d): value %.4f < LCL %.4f (Rule 1)",
                    batchIds[i], i, values[i], lcl));
            }
        }

        // Rule 2: 8 points in a row on one side of centre line
        int runAbove = 0, runBelow = 0;
        for (int i = 0; i < n; i++) {
            if (values[i] > mean) { runAbove++; runBelow = 0; }
            else if (values[i] < mean) { runBelow++; runAbove = 0; }
            else { runAbove = 0; runBelow = 0; }

            if (runAbove >= 8) {
                violations.add(String.format("Batch %s (idx %d): 8+ points above centre line (Rule 2)",
                    batchIds[i], i));
            }
            if (runBelow >= 8) {
                violations.add(String.format("Batch %s (idx %d): 8+ points below centre line (Rule 2)",
                    batchIds[i], i));
            }
        }

        return new ControlChart(metricCode, values, mr, mean, ucl, lcl, violations);
    }

    /**
     * Process Capability (Cpk) — only when both limits exist.
     */
    public record Capability(double cpk, double cp, double mean, double sigma,
                              double lowerSpec, double upperSpec) {}

    public static Capability computeCpk(double[] values, double lowerSpec, double upperSpec) {
        double mean = Arrays.stream(values).average().orElse(0);
        int n = values.length;
        double variance = 0;
        for (double v : values) variance += (v - mean) * (v - mean);
        double sigma = Math.sqrt(variance / (n - 1));

        double cpu = (upperSpec - mean) / (3 * sigma);
        double cpl = (mean - lowerSpec) / (3 * sigma);
        double cpk = Math.min(cpu, cpl);
        double cp = (upperSpec - lowerSpec) / (6 * sigma);

        return new Capability(cpk, cp, mean, sigma, lowerSpec, upperSpec);
    }

    /**
     * Compute summary metrics for a batch.
     */
    public static Map<String, Double> computeRunMetrics(
        double[] times, double[] vcd, double[] substrate, double[] byproduct
    ) {
        Map<String, Double> metrics = new LinkedHashMap<>();

        // Peak VCD
        metrics.put("peak_vcd", Arrays.stream(vcd).max().orElse(0));

        // Final values
        metrics.put("final_vcd", vcd[vcd.length - 1]);
        metrics.put("final_substrate", substrate[substrate.length - 1]);
        metrics.put("final_byproduct", byproduct[byproduct.length - 1]);

        // Integral of viable cells (trapezoidal)
        double ivc = 0;
        for (int i = 1; i < times.length; i++) {
            ivc += (vcd[i] + vcd[i - 1]) / 2 * (times[i] - times[i - 1]);
        }
        metrics.put("ivc", ivc);

        // Final viability (from VCD and assumed DCD — engine currently tracks xd separately)
        // Omitted if DCD not available.

        return metrics;
    }
}

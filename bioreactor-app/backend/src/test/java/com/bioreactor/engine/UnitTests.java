package com.bioreactor.engine;

import com.bioreactor.prediction.PredictionEngine;
import com.bioreactor.monitoring.MonitoringEngine;
import com.bioreactor.trending.TrendingEngine;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for monitoring, trending, and prediction — pure Java, no Spring, no DB.
 */
class UnitTests {

    // ── MonitoringEngine tests ──

    @Test
    void referenceBandDetectsOutsideBand() {
        double[] timeGrid = {0, 1, 2, 3, 4, 5};
        double[] p10 = {0.5, 1.0, 2.0, 3.0, 4.0, 5.0};
        double[] p50 = {1.0, 2.0, 4.0, 6.0, 8.0, 10.0};
        double[] p90 = {1.5, 3.0, 6.0, 9.0, 12.0, 15.0};

        var band = new MonitoringEngine.ReferenceBand("vcd", timeGrid, p10, p50, p90);

        // Value within band
        assertNull(band.check(2.0, 4.0), "Value at band center should be fine");

        // Value below p10
        String low = band.check(2.0, 1.0);
        assertNotNull(low, "Low value should trigger finding");
        assertTrue(low.contains("below 10th percentile"), "Message should mention below");

        // Value above p90
        String high = band.check(3.0, 12.0);
        assertNotNull(high, "High value should trigger finding");
        assertTrue(high.contains("above 90th percentile"), "Message should mention above");

        // Out of range time (before first grid point)
        assertNull(band.check(-0.5, 0.8), "Before grid should use first value");
        // Out of range time (after last grid point)
        assertNull(band.check(10.0, 5.0), "After grid should use last value");
    }

    @Test
    void buildBandPreservesOrdering() {
        // Two synthetic training runs
        List<double[]> timeSeries = List.of(
            new double[]{0, 1, 2, 3},
            new double[]{0, 1, 2, 3}
        );
        List<double[]> valueSeries = List.of(
            new double[]{0.5, 1.5, 2.5, 3.5},
            new double[]{1.5, 2.5, 3.5, 4.5}
        );
        double[] grid = {0, 1, 2, 3};

        var band = MonitoringEngine.buildBand("vcd", timeSeries, valueSeries, grid);

        assertEquals(4, band.p10().length);
        assertEquals(4, band.p50().length);
        assertEquals(4, band.p90().length);

        // p10 ≤ p50 ≤ p90 at every point
        for (int i = 0; i < grid.length; i++) {
            assertTrue(band.p10()[i] <= band.p50()[i],
                "p10 ≤ p50 at grid point " + i);
            assertTrue(band.p50()[i] <= band.p90()[i],
                "p50 ≤ p90 at grid point " + i);
        }
    }

    // ── TrendingEngine tests ──

    @Test
    void iMRChartDetectsViolations() {
        String[] batchIds = {"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"};
        double[] values = {95, 97, 96, 98, 94, 96, 97, 95, 96, 145}; // last is outlier

        var chart = TrendingEngine.buildIMRChart("peak_vcd", batchIds, values);

        assertNotNull(chart.centreLine());
        assertTrue(chart.ucl() > chart.centreLine(), "UCL should be above centre line");

        // Rule 1: the last point (145) should be beyond UCL
        boolean hasRule1 = chart.violations().stream()
            .anyMatch(v -> v.contains("Rule 1"));
        assertTrue(hasRule1, "Should detect out-of-control point (Rule 1)");

        // For Rule 2, run 8 points on same side of centre line
        // Use first point to bias mean up: mean = (200 + 8*50)/9 ≈ 66.7, so 50 < mean
        String[] ids2 = {"B0","B1","B2","B3","B4","B5","B6","B7","B8"};
        double[] vals2 = {200, 50, 50, 50, 50, 50, 50, 50, 50};
        var chart2 = TrendingEngine.buildIMRChart("metric", ids2, vals2);
        boolean hasRule2 = chart2.violations().stream()
            .anyMatch(v -> v.contains("Rule 2"));
        assertTrue(hasRule2, "Should detect 8-point run (Rule 2)");
    }

    @Test
    void capabilityCpkComputed() {
        double[] values = {95, 97, 96, 98, 94, 96, 97, 95, 96, 97};
        double lowerSpec = 90, upperSpec = 100;

        var cap = TrendingEngine.computeCpk(values, lowerSpec, upperSpec);

        assertTrue(cap.cpk() > 0, "Cpk should be positive");
        assertTrue(cap.cpk() > 1.0, "Cpk should show capable process");
        assertTrue(cap.cp() > 0, "Cp should be positive");
    }

    @Test
    void computeRunMetrics() {
        double[] times = {0, 1, 2, 3, 4};
        double[] vcd = {0.5, 1.5, 3.0, 4.0, 3.5};
        double[] sub = {10, 8, 5, 3, 1};
        double[] byp = {0, 0.5, 1.0, 1.5, 2.0};

        var metrics = TrendingEngine.computeRunMetrics(times, vcd, sub, byp);

        assertEquals(4.0, metrics.get("peak_vcd"), 0.01);
        assertEquals(3.5, metrics.get("final_vcd"), 0.01);
        assertEquals(1.0, metrics.get("final_substrate"), 0.01);
        assertEquals(2.0, metrics.get("final_byproduct"), 0.01);

        // IVC: trapezoidal integral of VCD
        // (0.5+1.5)/2*1 + (1.5+3)/2*1 + (3+4)/2*1 + (4+3.5)/2*1
        // = 1.0 + 2.25 + 3.5 + 3.75 = 10.5
        assertEquals(10.5, metrics.get("ivc"), 0.1);
    }

    // ── ModelParameter transform tests ──

    @Test
    void parameterTransformRoundTrip() {
        ModelParameter p = new ModelParameter("mu_max", 0.7, 0.01, 2.0, "1/h", "Growth rate");

        double start = 0.7;
        double unconstr = p.toUnconstrained(start);
        double back = p.fromUnconstrained(unconstr);
        assertEquals(start, back, 1e-9, "Round trip through unconstrained should be exact");

        // Mid-range
        double mid = 0.5;
        double um = p.toUnconstrained(mid);
        double bm = p.fromUnconstrained(um);
        assertEquals(mid, bm, 1e-9);

        // Near upper bound
        double nearUpper = 1.99;
        double uu = p.toUnconstrained(nearUpper);
        double bu = p.fromUnconstrained(uu);
        assertEquals(nearUpper, bu, 1e-6);
    }

    // ── Prediction band tests ──

    @Test
    void predictionBandFromErrors() {
        double[] times = {0, 1, 2, 3, 4};
        List<Double> errors = List.of(0.1, 0.2, 0.05, -0.15, 0.3);

        var band = PredictionEngine.PredictionBand.fromErrors("vcd", times, errors, 2.0);

        assertNotNull(band);
        assertEquals("vcd", band.parameterCode());

        // absError at each reference time should be positive
        for (int i = 0; i < times.length; i++) {
            double ae = band.absError(times[i]);
            assertTrue(ae >= 0, "absError should be non-negative at t=" + times[i]);
        }
    }
}

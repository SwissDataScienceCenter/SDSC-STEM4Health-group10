package com.bioreactor.engine;

import com.bioreactor.prediction.PredictionEngine;
import com.bioreactor.monitoring.MonitoringEngine;
import com.bioreactor.trending.TrendingEngine;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the full engine pipeline:
 * simulate → fit → predict → monitor → trend.
 *
 * Uses synthetic data from the textbook MonodModel. No challenge data.
 */
class EngineIntegrationTest {

    @Test
    void fullPipelineWithSyntheticData() {
        // 1. Generate synthetic training data
        var model = new MonodModel();
        var sim = new Simulator(model);
        var predEngine = new PredictionEngine(model);

        Map<String, Double> trueParams = Map.of(
            "mu_max", 0.70, "K_s", 0.02, "Y_XS", 0.50, "m_s", 0.025,
            "alpha", 0.0, "beta", 0.02, "k_d", 0.01, "k_lysis", 0.005
        );

        // Generate 10 synthetic training batches with different initial substrate
        List<Fitter.TrainingBatch> batches = new ArrayList<>();
        Random rng = new Random(123);
        for (int b = 0; b < 10; b++) {
            double sub0 = 4.0 + rng.nextDouble() * 2.0; // 4-6 g/L
            BioreactorState ic = new BioreactorState(
                0, 1.0, 0, sub0, 0, 0, 0.007, 10.0, 37.0, 7.0
            );
            var result = sim.simulate(ic, trueParams, List.of(), 14.0, 1.0);

            // Extract "measurements" at each output point + noise
            List<Double> times = new ArrayList<>();
            List<Double> vcd = new ArrayList<>();
            List<Double> tcd = new ArrayList<>();
            List<Double> sub = new ArrayList<>();
            List<Double> byp = new ArrayList<>();

            for (int i = 0; i < result.times().size(); i++) {
                double t = result.times().get(i);
                if (t < 0.01) continue; // skip t=0 (it's initial)
                BioreactorState s = result.states().get(i);
                times.add(t);
                vcd.add(s.xv() * (1 + rng.nextGaussian() * 0.03));
                tcd.add((s.xv() + s.xd()) * (1 + rng.nextGaussian() * 0.03));
                sub.add(Math.max(0, s.substrate() * (1 + rng.nextGaussian() * 0.05)));
                byp.add(Math.max(0, s.byproduct() * (1 + rng.nextGaussian() * 0.05)));
            }

            batches.add(new Fitter.TrainingBatch(
                "Batch" + (b + 1), ic, List.of(), times, vcd, tcd, sub, byp
            ));
        }

        // 2. Fit on all training batches
        var fitter = new Fitter(model);
        Fitter.FitResult fit = fitter.fit(batches, 100);

        assertTrue(fit.converged(), "Fit should converge");
        assertNotNull(fit.mapePerVariable().get("mape_vcd"));
        System.out.println("MAPE VCD: " + fit.mapePerVariable().get("mape_vcd"));

        // 3. Leave-one-out CV
        Map<String, Double> cvMape = fitter.crossValidate(batches, 50);
        assertTrue(cvMape.get("mape_vcd") < 100, "CV MAPE should be reasonable");
        System.out.println("CV MAPE VCD: " + cvMape.get("mape_vcd"));

        // 4. Prediction on a new "test" batch
        BioreactorState testIc = new BioreactorState(
            0, 1.0, 0, 5.0, 0, 0, 0.007, 10.0, 37.0, 7.0
        );

        List<Double> targetTimes = List.of(1.0, 3.0, 5.0, 7.0, 10.0, 14.0);
        var input = new PredictionEngine.PredictionInput(
            "TestBatch1", testIc, List.of(), targetTimes, 14.0
        );

        var pred = predEngine.predict(input, fit.fittedParams(), null);
        assertEquals("TestBatch1", pred.batchId());
        assertEquals(4 * targetTimes.size(), pred.rows().size(), // 4 params × N times
            "Should have 4 parameters x N time points");
        assertTrue(pred.warnings().size() > 0, "Should warn about missing bands");

        // 5. Build reference bands from training batches (using TRUE params for clean test)
        List<double[]> timeSeries = new ArrayList<>();
        List<double[]> valueSeries = new ArrayList<>();
        for (var batch : batches) {
            var simResult = sim.simulate(batch.initialState(), trueParams,
                batch.events(), 14.0, 0.5);
            double[] ts = simResult.times().stream().mapToDouble(d -> d).toArray();
            double[] vs = simResult.states().stream().mapToDouble(BioreactorState::xv).toArray();
            timeSeries.add(ts);
            valueSeries.add(vs);
        }
        double[] grid = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0};
        var band = MonitoringEngine.buildBand("vcd", timeSeries, valueSeries, grid);
        assertNotNull(band);
        assertEquals("vcd", band.parameterCode());

        // Verify band ordering: p10 <= p50 <= p90 at each grid point
        for (int i = 0; i < grid.length; i++) {
            assertTrue(band.p10()[i] <= band.p50()[i],
                "p10 <= p50 at t=" + grid[i]);
            assertTrue(band.p50()[i] <= band.p90()[i],
                "p50 <= p90 at t=" + grid[i]);
        }

        // 6. Monitoring: check a value against the band
        double checkVal = band.p90()[3] * 1.2; // 20% above 90th percentile
        String issue = band.check(5.0, checkVal);
        assertNotNull(issue, "Should detect value above 90th percentile");
        assertTrue(issue.contains("above"), issue);

        // Value below p10 should be detected
        double belowBand = band.p10()[3] * 0.5;
        String belowIssue = band.check(5.0, belowBand);
        assertNotNull(belowIssue, "Value well below p10 should trigger");
        assertTrue(belowIssue.contains("below"), belowIssue);

        // 7. Trending: I-MR chart
        double[] metrics = batches.stream()
            .mapToDouble(b -> b.observedVCD().stream().mapToDouble(v -> v).max().orElse(0))
            .toArray();
        String[] batchIds = batches.stream().map(Fitter.TrainingBatch::id).toArray(String[]::new);
        var chart = TrendingEngine.buildIMRChart("peak_vcd", batchIds, metrics);
        assertTrue(chart.ucl() > chart.lcl());
    }

    @Test
    void dcdScience() {
        // Verify: DCD = TCD - VCD
        BioreactorState s = new BioreactorState(1.0, 15.0, 3.0, 2.0, 0, 0.5, 0.007, 10.0, 37.0, 7.0);
        double tcd = s.xv() + s.xd(); // 18.0
        assertEquals(18.0, tcd, 0.001);
        // DCD is implicitly s.xd()
        assertEquals(3.0, tcd - s.xv(), 0.001);
    }

    @Test
    void predictionEngineOutputsCorrectRows() {
        var model = new MonodModel();
        var engine = new PredictionEngine(model);
        BioreactorState ic = new BioreactorState(
            0, 1.0, 0, 5.0, 0, 0, 0.007, 10.0, 37.0, 7.0
        );
        Map<String, Double> params = Map.of(
            "mu_max", 0.70, "K_s", 0.02, "Y_XS", 0.50, "m_s", 0.025,
            "alpha", 0.0, "beta", 0.02, "k_d", 0.01, "k_lysis", 0.005
        );

        var input = new PredictionEngine.PredictionInput(
            "Test", ic, List.of(), List.of(5.0), 10.0
        );
        var pred = engine.predict(input, params, null);

        // Should have 4 rows: vcd, tcd, substrate, byproduct at t=5.0
        assertEquals(4, pred.rows().size());
        Set<String> codes = new HashSet<>();
        for (var row : pred.rows()) codes.add(row.parameterCode());
        assertTrue(codes.containsAll(Set.of("vcd", "tcd", "substrate", "byproduct")));
    }
}

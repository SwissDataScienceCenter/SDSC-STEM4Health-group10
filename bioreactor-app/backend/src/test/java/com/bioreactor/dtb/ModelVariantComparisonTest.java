package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Head-to-head comparison of model variants on synthetic DTB-like data.
 *
 * Generates data from a "true" process, then fits each model variant
 * and reports MAPE (%) for VCD, TCD, Substrate, and Byproduct.
 */
class ModelVariantComparisonTest {

    // Ground truth params for synthetic data generation
    private static final Map<String, Double> TRUE_PARAMS;
    static {
        Map<String, Double> m = new HashMap<>();
        m.put("mu_max", 0.70); m.put("K_s", 0.02); m.put("K_i", 50.0);
        m.put("Y_XS", 0.50); m.put("m_s", 0.025); m.put("Y_BS", 0.90);
        m.put("alpha", 0.0); m.put("beta", 0.02); m.put("K_p", 50.0);
        m.put("k_d", 0.01); m.put("k_lysis", 0.005);
        m.put("feedB_S", 80.0); m.put("feedC_S", 0.0); m.put("feedD_S", 50.0);
        TRUE_PARAMS = Collections.unmodifiableMap(m);
    }

    @Test
    void compareAllModelsOnSyntheticData() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("  MODEL VARIANT COMPARISON — Synthetic DTB-like data with bolus feeds");
        System.out.println("=".repeat(80));

        // 1. Generate synthetic training data using the IMPROVED model as ground truth
        var trueModel = new DTBModelImproved();
        var sim = new Simulator(trueModel);

        List<Fitter.TrainingBatch> batches = new ArrayList<>();
        Random rng = new Random(42);

        for (int b = 0; b < 15; b++) {
            double sub0 = 3.0 + rng.nextDouble() * 3.0; // 3-6 g/L
            double vol0 = 8.0 + rng.nextDouble() * 4.0;  // 8-12 L
            BioreactorState ic = new BioreactorState(
                0, 0.5 + rng.nextDouble() * 1.5, 0, sub0, 0, 0.1, 0.007, vol0, 37.0, 7.0
            );

            // Generate random feed events (typical DTB pattern)
            List<Simulator.Event> events = new ArrayList<>();
            double tEnd = 11.0 + rng.nextDouble() * 3.0;
            int nFeeds = 2 + rng.nextInt(4); // 2-5 bolus feeds during the batch
            for (int f = 0; f < nFeeds; f++) {
                double feedTime = 0.5 + rng.nextDouble() * (tEnd - 1.0);
                double feedVol = 0.05 + rng.nextDouble() * 0.3; // 0.05-0.35 L
                events.add(Simulator.Event.feedBolus(feedTime, feedVol,
                    Map.of("substrate", TRUE_PARAMS.get("feedB_S"))));
                events.add(Simulator.Event.sampleVolume(feedTime,
                    ic.volume() + feedVol * (f + 1)));
            }

            var result = sim.simulate(ic, TRUE_PARAMS, events, tEnd, 0.5);

            // Extract "measurements" with realistic noise
            List<Double> times = new ArrayList<>();
            List<Double> vcd = new ArrayList<>();
            List<Double> tcd = new ArrayList<>();
            List<Double> sub = new ArrayList<>();
            List<Double> byp = new ArrayList<>();

            for (int i = 0; i < result.times().size(); i++) {
                double t = result.times().get(i);
                if (t < 0.01) continue; // skip t=0
                BioreactorState s = result.states().get(i);
                times.add(t);
                // 3% noise on VCD/TCD, 5% on substrate, 5% on byproduct
                vcd.add(s.xv() * (1 + rng.nextGaussian() * 0.03));
                tcd.add((s.xv() + s.xd()) * (1 + rng.nextGaussian() * 0.03));
                sub.add(Math.max(0, s.substrate() * (1 + rng.nextGaussian() * 0.05)));
                byp.add(Math.max(0, s.byproduct() * (1 + rng.nextGaussian() * 0.05)));
            }

            batches.add(new Fitter.TrainingBatch(
                "Batch" + (b + 1), ic, events, times, vcd, tcd, sub, byp
            ));
        }

        System.out.println("Generated " + batches.size() + " synthetic training batches");
        System.out.println("Ground truth: DTBModelImproved with Haldane + product inhibition");

        // 2. Fit all model variants and compare
        String[] modelNames = {"DTBModelImproved (this work)", "DTBV2Model (logistic+overflow)", "DTBModel (current)", "MonodModel (textbook)"};
        ProcessModel[] models = {new DTBModelImproved(), new DTBV2Model(), new DTBModel(), new MonodModel()};

        for (int m = 0; m < models.length; m++) {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("  Fitting: " + modelNames[m]);
            System.out.println("-".repeat(80));

            var fitter = new Fitter(models[m]);
            Fitter.FitResult fit = fitter.fitMultiStart(batches, 3, 150);

            System.out.println("  Converged: " + fit.converged());
            System.out.println("  Iterations: " + fit.iterations());
            System.out.println("  Final RMS cost: " + String.format("%.6f", fit.finalCost()));

            // Training MAPE
            System.out.println("\n  >>> TRAINING MAPE (%):");
            printMape("Training", fit.mapePerVariable());

            // Leave-one-batch-out CV
            System.out.println("\n  >>> CROSS-VALIDATION MAPE (%) [leave-one-batch-out]:");
            Map<String, Double> cvMape = fitter.crossValidate(batches, 100);
            printMape("CV", cvMape);

            // Print fitted parameters (key ones)
            System.out.println("\n  Fitted key parameters:");
            String[] keyParams = {"mu_max", "K_s", "K_i", "Y_XS", "m_s", "alpha", "beta", "K_p", "k_d",
                                  "Xmax", "kd0", "q_ovf", "Ks"};
            for (String kp : keyParams) {
                Double val = fit.fittedParams().get(kp);
                if (val != null) {
                    Double truth = TRUE_PARAMS.get(kp);
                    String mark = truth != null ? String.format(" (true=%.4f)", truth) : "";
                    System.out.printf("    %-12s = %.6f%s%n", kp, val, mark);
                }
            }
        }

        System.out.println("\n" + "=".repeat(80));
        System.out.println("  COMPARISON COMPLETE");
        System.out.println("=".repeat(80) + "\n");
    }

    @Test
    void testImprovedModelHasHaldaneAndInhibition() {
        DTBModelImproved improved = new DTBModelImproved();
        DTBModel original = new DTBModel();
        MonodModel monod = new MonodModel();

        assertEquals(14, improved.parameters().size(), "Improved model should have 14 params");
        assertEquals(9, original.parameters().size(), "Original DTB has 9 params");
        assertEquals(8, monod.parameters().size(), "Monod has 8 params");

        // Verify K_i and K_p exist in improved model
        Set<String> paramNames = new HashSet<>();
        improved.parameters().forEach(p -> paramNames.add(p.name()));
        assertTrue(paramNames.contains("K_i"), "Improved model must have K_i (Haldane)");
        assertTrue(paramNames.contains("K_p"), "Improved model must have K_p (product inhibition)");
        assertTrue(paramNames.contains("Y_BS"), "Improved model must have Y_BS (byproduct substrate yield)");
    }

    @Test
    void testProductInhibitionReducesGrowth() {
        var model = new DTBModelImproved();
        BioreactorState noInhib = new BioreactorState(1.0, 1.0, 0, 5.0, 0, 0.0, 0.007, 10.0, 37.0, 7.0);
        BioreactorState highByp = new BioreactorState(1.0, 1.0, 0, 5.0, 0, 50.0, 0.007, 10.0, 37.0, 7.0);

        double[] rateNo = model.rates(noInhib, TRUE_PARAMS);
        double[] rateHi = model.rates(highByp, TRUE_PARAMS);

        // With byproduct=50 and K_p=50, growth should be halved
        double ratio = rateHi[0] / Math.max(rateNo[0], 1e-9); // dXv ratio
        System.out.println("\nProduct inhibition test:");
        System.out.println("  dXv at B=0:  " + String.format("%.6f", rateNo[0]));
        System.out.println("  dXv at B=50: " + String.format("%.6f", rateHi[0]));
        System.out.println("  Ratio (expect ~0.5): " + String.format("%.4f", ratio));
        assertTrue(ratio < 0.7, "Product inhibition should significantly reduce growth rate");
    }

    @Test
    void testHaldaneInhibitionAtHighSubstrate() {
        var model = new DTBModelImproved();
        BioreactorState lowS = new BioreactorState(1.0, 1.0, 0, 5.0, 0, 0, 0.007, 10.0, 37.0, 7.0);
        BioreactorState highS = new BioreactorState(1.0, 1.0, 0, 100.0, 0, 0, 0.007, 10.0, 37.0, 7.0);

        double[] rateLow = model.rates(lowS, TRUE_PARAMS);
        double[] rateHigh = model.rates(highS, TRUE_PARAMS);

        double dXv_low = rateLow[0];
        double dXv_high = rateHigh[0];
        System.out.println("\nHaldane inhibition test:");
        System.out.println("  dXv at S=5:   " + String.format("%.6f", dXv_low));
        System.out.println("  dXv at S=100: " + String.format("%.6f", dXv_high));
        System.out.println("  dXv(high) < dXv(low)? " + (dXv_high < dXv_low));

        assertTrue(dXv_high < dXv_low,
            "Haldane should cause lower growth at substrate=100 than at substrate=5");
    }

    @Test
    void testDeathRateUsesSubstrateFactor() {
        var model = new DTBModelImproved();
        BioreactorState richS = new BioreactorState(1.0, 1.0, 0, 5.0, 0, 0, 0.007, 10.0, 37.0, 7.0);
        double[] rateRich = model.rates(richS, TRUE_PARAMS);

        BioreactorState poorS = new BioreactorState(1.0, 1.0, 0, 0.001, 0, 0, 0.007, 10.0, 37.0, 7.0);
        double[] ratePoor = model.rates(poorS, TRUE_PARAMS);

        double netXd_rich = rateRich[1]; // dXd
        double netXd_poor = ratePoor[1];

        System.out.println("\nDeath rate test:");
        System.out.println("  dXd at S=5:     " + String.format("%.6f", netXd_rich));
        System.out.println("  dXd at S=0.001: " + String.format("%.6f", netXd_poor));
        System.out.println("  dXd(poor) > dXd(rich)? " + (netXd_poor > netXd_rich));

        assertTrue(netXd_poor > netXd_rich,
            "Death rate should be higher at low substrate");
    }

    private static void printMape(String label, Map<String, Double> mape) {
        String[] keys = {"mape_vcd", "mape_tcd", "mape_substrate", "mape_byproduct"};
        String[] labels = {"VCD", "TCD", "Substrate", "Byproduct"};
        double sum = 0;
        int count = 0;
        for (int i = 0; i < keys.length; i++) {
            Double v = mape.get(keys[i]);
            if (v != null && !v.isNaN()) {
                System.out.printf("    %-15s = %6.2f%%%n", labels[i], v);
                sum += v;
                count++;
            }
        }
        if (count > 0) {
            System.out.printf("    %-15s = %6.2f%%%n", "MEAN", sum / count);
        }
    }
}

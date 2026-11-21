package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Head-to-head comparison of model variants on synthetic DTB-like data.
 *
 * Uses the DTBModelAdvanced (threshold inhibition + byproduct consumption)
 * as ground truth to generate training data that mimics real DTB behavior:
 * - VCD grows exponentially then decelerates
 * - Substrate consumed, replenished by bolus feeds
 * - Byproduct rises then is consumed in late phase (as seen in real data!)
 * - Only Feed B carries substrate (C and D are pH control / trace, confirmed by data)
 */
class ModelVariantComparisonTest {

    // Ground truth params for the Advanced model
    private static final Map<String, Double> TRUE_PARAMS;
    static {
        Map<String, Double> m = new HashMap<>();
        m.put("mu_max", 0.70); m.put("K_s", 0.02); m.put("K_i", 50.0);
        m.put("B_thresh", 1.0); m.put("B_crit", 5.0);
        m.put("Y_XS", 0.50); m.put("m_s", 0.025);
        m.put("alpha", 0.0); m.put("beta", 0.02);
        m.put("q_B_max", 0.08); m.put("K_Bp", 1.0); m.put("K_sB", 0.5); m.put("Y_XB", 0.40);
        m.put("k_d", 0.01); m.put("k_lysis", 0.005);
        m.put("feedB_S", 120.0);
        TRUE_PARAMS = Collections.unmodifiableMap(m);
    }

    @Test
    void compareAllModelsOnSyntheticData() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("  MODEL VARIANT COMPARISON — 5 models on realistic DTB synthetic data");
        System.out.println("  Ground truth: DTBModelAdvanced (threshold inhib + byproduct consumption)");
        System.out.println("=".repeat(80));

        var truthModel = new DTBModelAdvanced();
        var sim = new Simulator(truthModel);
        List<Fitter.TrainingBatch> batches = new ArrayList<>();
        Random rng = new Random(42);

        for (int b = 0; b < 15; b++) {
            double sub0 = 3.5 + rng.nextDouble() * 2.5;
            double vol0 = 9.0 + rng.nextDouble() * 3.0;
            BioreactorState ic = new BioreactorState(
                0, 0.5 + rng.nextDouble() * 1.5, 0, sub0, 0, 0.1, 0.007, vol0, 37.0, 7.0
            );

            List<Simulator.Event> events = new ArrayList<>();
            double tEnd = 10.0 + rng.nextDouble() * 3.0;
            int nFeeds = 3 + rng.nextInt(4);
            for (int f = 0; f < nFeeds; f++) {
                double feedTime = 1.0 + rng.nextDouble() * (tEnd - 2.0);
                double feedVol = 0.05 + rng.nextDouble() * 0.25;
                events.add(Simulator.Event.feedBolus(feedTime, feedVol,
                    Map.of("substrate", TRUE_PARAMS.get("feedB_S"))));
                events.add(Simulator.Event.sampleVolume(feedTime,
                    ic.volume() + feedVol * (f + 1)));
            }

            var result = sim.simulate(ic, TRUE_PARAMS, events, tEnd, 0.5);

            List<Double> times = new ArrayList<>();
            List<Double> vcd = new ArrayList<>();
            List<Double> tcd = new ArrayList<>();
            List<Double> sub = new ArrayList<>();
            List<Double> byp = new ArrayList<>();

            for (int i = 0; i < result.times().size(); i++) {
                double t = result.times().get(i);
                if (t < 0.01) continue;
                BioreactorState s = result.states().get(i);
                times.add(t);
                vcd.add(Math.max(0, s.xv() * (1 + rng.nextGaussian() * 0.03)));
                tcd.add(Math.max(0, (s.xv() + s.xd()) * (1 + rng.nextGaussian() * 0.03)));
                sub.add(Math.max(0, s.substrate() * (1 + rng.nextGaussian() * 0.05)));
                byp.add(Math.max(0, s.byproduct() * (1 + rng.nextGaussian() * 0.05)));
            }

            batches.add(new Fitter.TrainingBatch(
                "Batch" + (b + 1), ic, events, times, vcd, tcd, sub, byp
            ));
        }

        System.out.println("Generated " + batches.size() + " synthetic training batches\n");

        String[] names = {
            "DTBModelAdvanced (threshold+byp-consume)",
            "DTBModelImproved (Haldane+Kp)",
            "DTBV2Model (logistic+overflow)",
            "DTBModel (current)",
            "MonodModel (textbook)"
        };
        ProcessModel[] models = {
            new DTBModelAdvanced(), new DTBModelImproved(),
            new DTBV2Model(), new DTBModel(), new MonodModel()
        };

        List<String[]> summary = new ArrayList<>();
        summary.add(new String[]{"Model", "VCD%", "TCD%", "Sub%", "Byp%", "Mean%", "RMS"});

        for (int m = 0; m < models.length; m++) {
            System.out.println("-".repeat(80));
            System.out.println("  Fitting: " + names[m]);
            System.out.println("-".repeat(80));

            var fitter = new Fitter(models[m]);
            Fitter.FitResult fit = fitter.fitMultiStart(batches, 3, 200);

            System.out.println("  Converged: " + fit.converged() +
                "  Iterations: " + fit.iterations() +
                "  RMS: " + String.format("%.4f", fit.finalCost()));

            // CV MAPE
            Map<String, Double> cv = fitter.crossValidate(batches, 150);
            System.out.println("\n  >>> CROSS-VALIDATION MAPE (%):");
            String[] labels = {"VCD", "TCD", "Substrate", "Byproduct"};
            double[] vals = new double[4];
            double sumV = 0; int count = 0;
            for (int i = 0; i < 4; i++) {
                String key = "mape_" + labels[i].toLowerCase();
                Double v = cv.get(key);
                vals[i] = (v != null && !v.isNaN()) ? v : Double.NaN;
                if (!Double.isNaN(vals[i])) {
                    System.out.printf("    %-15s = %6.2f%%%n", labels[i], vals[i]);
                    sumV += vals[i]; count++;
                }
            }
            double meanV = count > 0 ? sumV / count : Double.NaN;
            System.out.printf("    %-15s = %6.2f%%%n", "MEAN", meanV);

            summary.add(new String[]{
                names[m].substring(0, Math.min(38, names[m].length())),
                String.format("%.2f", vals[0]),
                String.format("%.2f", vals[1]),
                String.format("%.2f", vals[2]),
                String.format("%.2f", vals[3]),
                String.format("%.2f", meanV),
                String.format("%.4f", fit.finalCost())
            });
        }

        // Print summary table
        System.out.println("\n" + "=".repeat(80));
        System.out.println("  FINAL SUMMARY — Cross-Validation MAPE (%):");
        System.out.println("=".repeat(80));
        for (String[] row : summary) {
            System.out.printf("  %-38s %7s %7s %7s %7s %7s %8s%n",
                row[0], row[1], row[2], row[3], row[4], row[5], row[6]);
        }
        System.out.println();
    }

    @Test
    void testAdvancedModelSanity() {
        var model = new DTBModelAdvanced();
        assertEquals(16, model.parameters().size());

        Set<String> names = new HashSet<>();
        model.parameters().forEach(p -> names.add(p.name()));
        assertTrue(names.contains("B_thresh"), "Must have B_thresh");
        assertTrue(names.contains("q_B_max"), "Must have q_B_max");
        assertTrue(names.contains("Y_XB"), "Must have Y_XB");
        assertTrue(names.contains("K_sB"), "Must have K_sB");
        assertTrue(names.contains("K_i"), "Must have K_i (Haldane)");
    }

    @Test
    void testThresholdInhibition() {
        var model = new DTBModelAdvanced();
        Map<String, Double> p = new HashMap<>(TRUE_PARAMS);

        BioreactorState lowB = new BioreactorState(1.0, 1.0, 0, 5.0, 0, 0.5, 0.007, 10.0, 37.0, 7.0);
        double[] rLow = model.rates(lowB, p);

        BioreactorState midB = new BioreactorState(1.0, 1.0, 0, 5.0, 0, 3.0, 0.007, 10.0, 37.0, 7.0);
        double[] rMid = model.rates(midB, p);

        BioreactorState critB = new BioreactorState(1.0, 1.0, 0, 5.0, 0, 5.0, 0.007, 10.0, 37.0, 7.0);
        double[] rCrit = model.rates(critB, p);

        System.out.println("\nThreshold inhibition test:");
        System.out.printf("  dXv at B=0.5 (below thresh): %.6f%n", rLow[0]);
        System.out.printf("  dXv at B=3.0 (mid inhib):    %.6f%n", rMid[0]);
        System.out.printf("  dXv at B=5.0 (at crit):     %.6f%n", rCrit[0]);

        assertTrue(rLow[0] > rMid[0], "Growth should decrease with byproduct");
        assertTrue(rCrit[0] < 0.01, "Growth near zero at critical byproduct");
    }

    @Test
    void testByproductConsumptionAtLowSubstrate() {
        var model = new DTBModelAdvanced();
        Map<String, Double> p = new HashMap<>(TRUE_PARAMS);

        BioreactorState depleted = new BioreactorState(1.0, 1.0, 0, 0.01, 0, 2.0, 0.007, 10.0, 37.0, 7.0);
        double[] rDep = model.rates(depleted, p);

        BioreactorState plenty = new BioreactorState(1.0, 1.0, 0, 5.0, 0, 2.0, 0.007, 10.0, 37.0, 7.0);
        double[] rPlenty = model.rates(plenty, p);

        BioreactorState bothLow = new BioreactorState(1.0, 1.0, 0, 0.01, 0, 0.01, 0.007, 10.0, 37.0, 7.0);
        double[] rBoth = model.rates(bothLow, p);

        System.out.println("\nByproduct consumption test:");
        System.out.printf("  S=0.01 B=2.0:  dXv=%.6f dS=%.6f dB=%.6f%n", rDep[0], rDep[2], rDep[4]);
        System.out.printf("  S=5.0  B=2.0:  dXv=%.6f dS=%.6f dB=%.6f%n", rPlenty[0], rPlenty[2], rPlenty[4]);
        System.out.printf("  S=0.01 B=0.01: dXv=%.6f dS=%.6f dB=%.6f%n", rBoth[0], rBoth[2], rBoth[4]);

        // Key: dB should be NEGATIVE when S depleted and B available (consuming byproduct)
        assertTrue(rDep[4] < 0,
            "dB should be negative when substrate depleted. dB=" + rDep[4]);

        // When S high, byproduct consumption suppressed, dB should be positive
        assertTrue(rPlenty[4] > 0,
            "dB should be positive when substrate is abundant. dB=" + rPlenty[4]);
    }

    @Test
    void testHaldaneInhibitionAtHighSubstrate() {
        var model = new DTBModelAdvanced();
        Map<String, Double> p = new HashMap<>(TRUE_PARAMS);

        BioreactorState lowS = new BioreactorState(1.0, 1.0, 0, 5.0, 0, 0, 0.007, 10.0, 37.0, 7.0);
        double[] rateLow = model.rates(lowS, p);

        BioreactorState highS = new BioreactorState(1.0, 1.0, 0, 100.0, 0, 0, 0.007, 10.0, 37.0, 7.0);
        double[] rateHigh = model.rates(highS, p);

        System.out.println("\nHaldane inhibition test:");
        System.out.printf("  dXv at S=5:   %.6f%n", rateLow[0]);
        System.out.printf("  dXv at S=100: %.6f%n", rateHigh[0]);

        assertTrue(rateHigh[0] < rateLow[0],
            "Haldane: growth at S=100 should be LESS than at S=5");
    }

    @Test
    void testDeathRateUsesSubstrateFactor() {
        var model = new DTBModelAdvanced();
        Map<String, Double> p = new HashMap<>(TRUE_PARAMS);

        BioreactorState richS = new BioreactorState(1.0, 1.0, 0, 5.0, 0, 0, 0.007, 10.0, 37.0, 7.0);
        double[] rateRich = model.rates(richS, p);

        BioreactorState poorS = new BioreactorState(1.0, 1.0, 0, 0.001, 0, 0, 0.007, 10.0, 37.0, 7.0);
        double[] ratePoor = model.rates(poorS, p);

        System.out.println("\nDeath rate test:");
        System.out.printf("  dXd at S=5:     %.6f%n", rateRich[1]);
        System.out.printf("  dXd at S=0.001: %.6f%n", ratePoor[1]);

        assertTrue(ratePoor[1] > rateRich[1],
            "Death should be higher at low substrate");
    }
}

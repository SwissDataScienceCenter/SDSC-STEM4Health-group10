package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.util.*;

/**
 * Head-to-head comparison of DTBV3Model vs DTBV3GIModel on real DTB data.
 *
 * Tests whether the added growth-inhibition-by-byproduct term (K_IP)
 * improves holdout MAPE (especially byproduct).
 */
class V3GIModelComparisonTest {

    private static final String CSV_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv";

    @Test
    void compareV3AndV3GI() throws Exception {
        var csvReader = new DTBCsvReader(CSV_PATH);
        List<DTBCsvReader.DTBBatch> all = csvReader.loadAllBatches();

        List<DTBCsvReader.DTBBatch> train = all.stream().filter(b -> !b.isTest()).toList();
        List<DTBCsvReader.DTBBatch> test = all.stream().filter(DTBCsvReader.DTBBatch::isTest).toList();

        System.out.println("\n" + "=".repeat(80));
        System.out.println("  V3 vs V3-GI MODEL COMPARISON");
        System.out.println("  Training: " + train.size() + " | Test: " + test.size());
        System.out.println("=".repeat(80));

        // Test both models
        ProcessModel[] models = {new DTBV3Model(), new DTBV3GIModel()};
        String[] names = {"V3 (no growth inhibition)", "V3-GI (with K_IP growth inhibition)"};

        for (int mi = 0; mi < models.length; mi++) {
            ProcessModel model = models[mi];
            var fitter = new Fitter(model);

            // Convert training batches
            List<Fitter.TrainingBatch> fitBatches = new ArrayList<>();
            for (var b : train) {
                BioreactorState ic = new BioreactorState(0, b.initialState().xv(), 0,
                    b.initialState().substrate(), 0, b.initialState().byproduct(),
                    0.007, b.initialState().volume(), 37.0, 7.0,
                    model.defaultHiddenState());
                fitBatches.add(new Fitter.TrainingBatch(b.batchId(), ic, b.events(),
                    b.measurementTimes(),
                    b.measurements().getOrDefault("vcd", List.of()),
                    b.measurements().getOrDefault("tcd", List.of()),
                    b.measurements().getOrDefault("substrate", List.of()),
                    b.measurements().getOrDefault("byproduct", List.of())));
            }

            // Shuffle for train/holdout split
            Collections.shuffle(fitBatches, new Random(42));
            int nHold = Math.max(5, fitBatches.size() / 10);
            List<Fitter.TrainingBatch> ct = fitBatches.subList(0, fitBatches.size() - nHold);
            List<Fitter.TrainingBatch> ch = fitBatches.subList(fitBatches.size() - nHold, fitBatches.size());

            System.out.println("\n  --- " + names[mi] + " ---");
            System.out.println("  Fit on: " + ct.size() + " | Holdout: " + ch.size());

            System.out.println("  Fitting " + model.parameters().size() + " params, 500 evals...");
            long startMs = System.currentTimeMillis();
            Fitter.FitResult fit = fitter.fit(ct, 500);
            long elapsed = (System.currentTimeMillis() - startMs) / 1000;

            System.out.println("  Time: " + elapsed + "s | Converged: " + fit.converged() +
                " | Iter: " + fit.iterations() + " | RMS: " + String.format("%.6f", fit.finalCost()));

            // Key fitted params
            System.out.println("  Key fitted params:");
            for (String key : List.of("mu_max", "Ks", "K_N", "Yxn", "K_IP", "K_iP",
                    "Yps", "kd_N", "kd_D", "k_r", "C_b")) {
                Double val = fit.fittedParams().get(key);
                if (val != null) {
                    System.out.printf("    %-12s = %12.6f  (start: %12.6f)%n",
                        key, val, model.parameters().stream()
                            .filter(mp -> mp.name().equals(key)).findFirst()
                            .map(ModelParameter::startValue).orElse(Double.NaN));
                }
            }

            // MAPE on holdout
            double[] sums = new double[4];
            int[] counts = new int[4];
            for (var hb : ch) {
                var pred = fitter.predict(hb, fit.fittedParams());
                for (int i = 1; i < hb.times().size(); i++) {
                    double t = hb.times().get(i);
                    var sim = nearestState(pred, t);

                    double oVcd = i < hb.observedVCD().size() ? hb.observedVCD().get(i) : Double.NaN;
                    double oTcd = i < hb.observedTCD().size() ? hb.observedTCD().get(i) : Double.NaN;
                    double oSub = i < hb.observedSubstrate().size() ? hb.observedSubstrate().get(i) : Double.NaN;
                    double oByp = i < hb.observedByproduct().size() ? hb.observedByproduct().get(i) : Double.NaN;

                    double[] obs = {oVcd, oTcd, oSub, oByp};
                    double[] preds = {sim.xv(), sim.xv() + sim.xd(),
                        Math.max(0, sim.substrate()), Math.max(0, sim.byproduct())};

                    for (int j = 0; j < 4; j++) {
                        if (!Double.isNaN(obs[j]) && obs[j] > 1e-6) {
                            sums[j] += Math.abs((preds[j] - obs[j]) / obs[j]);
                            counts[j]++;
                        }
                    }
                }
            }

            System.out.println("\n  HOLDOUT MAPE:");
            String[] labels = {"VCD", "TCD", "Substrate", "Byproduct"};
            double meanMape = 0;
            int cnt = 0;
            for (int j = 0; j < 4; j++) {
                double mape = counts[j] > 0 ? sums[j] / counts[j] * 100 : Double.NaN;
                System.out.printf("    %-15s = %7.2f%%  (n=%d)%n", labels[j], mape, counts[j]);
                if (!Double.isNaN(mape)) { meanMape += mape; cnt++; }
            }
            meanMape = cnt > 0 ? meanMape / cnt : Double.NaN;
            System.out.printf("    %-15s = %7.2f%%%n", "MEAN", meanMape);
        }

        System.out.println("\n" + "=".repeat(80));
        System.out.println("  COMPARISON COMPLETE");
        System.out.println("=".repeat(80) + "\n");
    }

    private static BioreactorState nearestState(RK4Simulator.SimResult r, double t) {
        int idx = 0;
        for (int i = 0; i < r.times().size(); i++) {
            if (r.times().get(i) >= t) { idx = i; break; }
            idx = i;
        }
        return r.states().get(idx);
    }
}

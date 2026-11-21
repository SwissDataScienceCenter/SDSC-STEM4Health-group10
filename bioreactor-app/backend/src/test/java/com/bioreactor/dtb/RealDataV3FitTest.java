package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Fits DTBV3Model on REAL DTB data and reports MAPE.
 *
 * This is a focused test that loads the CSV, fits on an 80/20 train/holdout split,
 * and prints results. Ported from Python mechanistic.py + evaluation.py pipeline.
 */
class RealDataV3FitTest {

    private static final String CSV_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv";

    @Test
    void fitV3OnRealData() throws Exception {
        var csvReader = new DTBCsvReader(CSV_PATH);
        List<DTBCsvReader.DTBBatch> all = csvReader.loadAllBatches();

        // Split: training (to_predict=false) and test (to_predict=true)
        List<DTBCsvReader.DTBBatch> train = all.stream().filter(b -> !b.isTest()).toList();
        List<DTBCsvReader.DTBBatch> test = all.stream().filter(DTBCsvReader.DTBBatch::isTest).toList();

        System.out.println("\n" + "=".repeat(80));
        System.out.println("  DTB V3 MODEL FIT ON REAL DATA");
        System.out.println("  Training batches: " + train.size() + " | Test batches: " + test.size());
        System.out.println("=".repeat(80));

        // Convert to Fitter training batches
        List<Fitter.TrainingBatch> fitBatches = new ArrayList<>();
        for (var b : train) {
            fitBatches.add(toFitBatch(b));
        }

        // Use a small subset for fast iteration (5 fit, 3 holdout)
        Collections.shuffle(fitBatches, new Random(42));
        List<Fitter.TrainingBatch> fitTrain = fitBatches.subList(0, Math.min(5, fitBatches.size()));
        List<Fitter.TrainingBatch> fitHold = fitBatches.subList(
            Math.min(5, fitBatches.size()),
            Math.min(8, fitBatches.size()));

        System.out.println("  Fit on: " + fitTrain.size() + " | Holdout: " + fitHold.size());

        var model = new DTBV3Model();
        var fitter = new Fitter(model);

        System.out.println("\n  Fitting V3 (22 params), single start, 100 function evals...");
        long startMs = System.currentTimeMillis();
        Fitter.FitResult fit = fitter.fit(fitTrain, 100);
        long elapsed = (System.currentTimeMillis() - startMs) / 1000;

        System.out.println("  Time: " + elapsed + "s | Converged: " + fit.converged() +
            " | Iter: " + fit.iterations() + " | RMS: " + String.format("%.6f", fit.finalCost()));

        // Report fitted params
        System.out.println("\n  FITTED PARAMETERS:");
        List<String> keyParams = List.of(
            "mu_max", "Ks", "K_N", "Yxn", "Yxs", "m", "kd0", "kd_N", "k_lys",
            "q_ovf", "K_ovf", "n_ovf", "K_iP", "Yps", "qPc", "K_Pc", "k_r", "kd_D"
        );
        for (String key : keyParams) {
            Double val = fit.fittedParams().get(key);
            Double startVal = fit.startParams().get(key);
            if (val != null) {
                System.out.printf("    %-12s = %12.6f  (start: %12.6f)%n", key, val, startVal);
            }
        }

        // MAPE on holdout
        System.out.println("\n  HOLDOUT MAPE:");
        double[] sums = new double[4];
        int[] counts = new int[4];
        String[] labels = {"VCD", "TCD", "Substrate", "Byproduct"};

        for (var hb : fitHold) {
            var pred = fitter.predict(hb, fit.fittedParams());
            for (int i = 1; i < hb.times().size(); i++) {
                double t = hb.times().get(i);
                var sim = nearestState(pred, t);

                double oVcd = i < hb.observedVCD().size() ? hb.observedVCD().get(i) : Double.NaN;
                double oTcd = i < hb.observedTCD().size() ? hb.observedTCD().get(i) : Double.NaN;
                double oSub = i < hb.observedSubstrate().size() ? hb.observedSubstrate().get(i) : Double.NaN;
                double oByp = i < hb.observedByproduct().size() ? hb.observedByproduct().get(i) : Double.NaN;

                double[] obs = {oVcd, oTcd, oSub, oByp};
                double[] preds = {sim.xv(), sim.xv() + sim.xd(), Math.max(0, sim.substrate()), Math.max(0, sim.byproduct())};

                for (int j = 0; j < 4; j++) {
                    if (!Double.isNaN(obs[j]) && obs[j] > 1e-6) {
                        sums[j] += Math.abs((preds[j] - obs[j]) / obs[j]);
                        counts[j]++;
                    }
                }
            }
        }

        double meanMape = 0;
        int cnt = 0;
        for (int j = 0; j < 4; j++) {
            double mape = counts[j] > 0 ? sums[j] / counts[j] * 100 : Double.NaN;
            System.out.printf("    %-15s = %7.2f%%  (n=%d)%n", labels[j], mape, counts[j]);
            if (!Double.isNaN(mape)) { meanMape += mape; cnt++; }
        }
        meanMape = cnt > 0 ? meanMape / cnt : Double.NaN;
        System.out.printf("    %-15s = %7.2f%%%n", "MEAN", meanMape);

        // Reference values
        System.out.println("\n  REFERENCE (Python V3, 10-fold CV on 49 train):");
        System.out.println("    VCD=8.6%  TCD=8.5%  Substrate=19.8%  Byproduct=18.5%  Mean=13.8%");
        System.out.println("=".repeat(80) + "\n");
    }

    private Fitter.TrainingBatch toFitBatch(DTBCsvReader.DTBBatch b) {
        var vcd = b.measurements().getOrDefault("vcd", List.of());
        var tcd = b.measurements().getOrDefault("tcd", List.of());
        var sub = b.measurements().getOrDefault("substrate", List.of());
        var byp = b.measurements().getOrDefault("byproduct", List.of());
        return new Fitter.TrainingBatch(
            b.batchId(), b.initialState(), b.events(),
            b.measurementTimes(), vcd, tcd, sub, byp
        );
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

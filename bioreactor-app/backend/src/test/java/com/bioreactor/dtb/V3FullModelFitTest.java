package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.util.*;

/**
 * Quick MAPE check on V3 full model with scaled residuals.
 */
class V3FullModelFitTest {

    private static final String CSV_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv";

    @Test
    void fitV3FullWithScaledResiduals() throws Exception {
        var csvReader = new DTBCsvReader(CSV_PATH);
        List<DTBCsvReader.DTBBatch> all = csvReader.loadAllBatches();
        List<DTBCsvReader.DTBBatch> train = all.stream().filter(b -> !b.isTest()).toList();

        var model = new DTBV3Model();
        var fitter = new Fitter(model);

        List<Fitter.TrainingBatch> batches = new ArrayList<>();
        for (var b : train) {
            BioreactorState ic = new BioreactorState(0, b.initialState().xv(), 0,
                b.initialState().substrate(), 0, b.initialState().byproduct(),
                0.007, b.initialState().volume(), 37.0, 7.0,
                model.defaultHiddenState());
            batches.add(new Fitter.TrainingBatch(b.batchId(), ic, b.events(),
                b.measurementTimes(), b.measurements().getOrDefault("vcd", List.of()),
                b.measurements().getOrDefault("tcd", List.of()),
                b.measurements().getOrDefault("substrate", List.of()),
                b.measurements().getOrDefault("byproduct", List.of())));
        }

        // Shuffle + split
        Collections.shuffle(batches, new Random(42));
        int nHold = Math.max(5, batches.size() / 10);
        var ct = batches.subList(0, batches.size() - nHold);
        var ch = batches.subList(batches.size() - nHold, batches.size());

        System.out.println("\n  --- V3 Full (22 free params, scaled residuals) ---");
        System.out.println("  Fit: " + ct.size() + " | Holdout: " + ch.size());

        long startMs = System.currentTimeMillis();
        Fitter.FitResult fit = fitter.fit(ct, 2000);
        long elapsed = (System.currentTimeMillis() - startMs) / 1000;

        System.out.println("  Time: " + elapsed + "s | Converged: " + fit.converged() +
            " | Iter: " + fit.iterations() + " | RMS: " + String.format("%.6f", fit.finalCost()));

        // Key params
        for (String key : List.of("mu_max", "Ks", "K_N", "Yxn", "kd_N", "kd_D", "k_lys",
                "Yxs", "q_ovf", "K_iP", "Yps", "C_b", "C_sub", "K_Pc")) {
            Double val = fit.fittedParams().get(key);
            if (val != null)
                System.out.printf("    %-12s = %12.6f  (start: %12.6f)%n",
                    key, val, model.parameters().stream()
                        .filter(mp -> mp.name().equals(key)).findFirst()
                        .map(ModelParameter::startValue).orElse(Double.NaN));
        }

        // MAPE
        double[] sums = new double[4]; int[] counts = new int[4];
        String[] labels = {"VCD", "TCD", "Substrate", "Byproduct"};
        for (var hb : ch) {
            var pred = fitter.predict(hb, fit.fittedParams());
            for (int i = 1; i < hb.times().size(); i++) {
                double t = hb.times().get(i);
                var sim = nearestState(pred, t);
                double[] obs = {
                    i < hb.observedVCD().size() ? hb.observedVCD().get(i) : Double.NaN,
                    i < hb.observedTCD().size() ? hb.observedTCD().get(i) : Double.NaN,
                    i < hb.observedSubstrate().size() ? hb.observedSubstrate().get(i) : Double.NaN,
                    i < hb.observedByproduct().size() ? hb.observedByproduct().get(i) : Double.NaN
                };
                double[] preds = {sim.xv(), sim.xv() + sim.xd(), Math.max(0, sim.substrate()), Math.max(0, sim.byproduct())};
                for (int j = 0; j < 4; j++) {
                    if (!Double.isNaN(obs[j]) && obs[j] > 1e-6) {
                        sums[j] += Math.abs((preds[j] - obs[j]) / obs[j]);
                        counts[j]++;
                    }
                }
            }
        }
        System.out.println("\n  HOLDOUT MAPE:");
        double mean = 0; int n = 0;
        for (int j = 0; j < 4; j++) {
            double m = counts[j] > 0 ? sums[j] / counts[j] * 100 : Double.NaN;
            System.out.printf("    %-15s = %7.2f%%  (n=%d)%n", labels[j], m, counts[j]);
            if (!Double.isNaN(m)) { mean += m; n++; }
        }
        System.out.printf("    %-15s = %7.2f%%%n", "MEAN", n > 0 ? mean / n : Double.NaN);
        System.out.println("  Python V3 ref: VCD=8.6% TCD=8.5% Sub=19.8% Byp=18.5% Mean=13.8%");
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

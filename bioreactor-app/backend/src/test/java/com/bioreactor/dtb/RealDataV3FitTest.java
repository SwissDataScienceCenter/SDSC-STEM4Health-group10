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

        System.out.println("\n  --- V3 Constrained (12 params) ---");
        // Use constrained model (12 free params + 10 locked to literature)
        var cModel = new DTBV3ConstrainedModel();
        var cFitter = new Fitter(cModel);
        List<Fitter.TrainingBatch> cBatches = new ArrayList<>();
        for (var b : train) {
            BioreactorState ic = new BioreactorState(0, b.initialState().xv(), 0,
                b.initialState().substrate(), 0, b.initialState().byproduct(),
                0.007, b.initialState().volume(), 37.0, 7.0,
                cModel.defaultHiddenState());
            cBatches.add(new Fitter.TrainingBatch(b.batchId(), ic, b.events(),
                b.measurementTimes(), b.measurements().getOrDefault("vcd", List.of()),
                b.measurements().getOrDefault("tcd", List.of()),
                b.measurements().getOrDefault("substrate", List.of()),
                b.measurements().getOrDefault("byproduct", List.of())));
        }
        // Fit on all 49 training batches, 10% holdout for validation
        Collections.shuffle(cBatches, new Random(42));
        int nHold = Math.max(5, cBatches.size() / 10);
        List<Fitter.TrainingBatch> ct = cBatches.subList(0, cBatches.size() - nHold);
        List<Fitter.TrainingBatch> ch = cBatches.subList(cBatches.size() - nHold, cBatches.size());
        System.out.println("  Fit on: " + ct.size() + " | Holdout: " + ch.size());

        System.out.println("  Fitting constrained (12 params), single start, 1000 evals...");
        long startMs = System.currentTimeMillis();
        Fitter.FitResult fit = cFitter.fit(ct, 1000);
        long elapsed = (System.currentTimeMillis() - startMs) / 1000;

        System.out.println("  Time: " + elapsed + "s | Converged: " + fit.converged() +
            " | Iter: " + fit.iterations() + " | RMS: " + String.format("%.6f", fit.finalCost()));

        // Report fitted params
        System.out.println("\n  FITTED PARAMETERS:");
        for (var mp : cModel.parameters()) {
            Double val = fit.fittedParams().get(mp.name());
            if (val != null) {
                System.out.printf("    %-12s = %12.6f  (start: %12.6f)%n", mp.name(), val, mp.startValue());
            }
        }

        // MAPE on holdout
        System.out.println("\n  HOLDOUT MAPE:");
        double[] sums = new double[4];
        int[] counts = new int[4];
        String[] labels = {"VCD", "TCD", "Substrate", "Byproduct"};

        for (var hb : ch) {
            var pred = cFitter.predict(hb, fit.fittedParams());
            for (int i = 1; i < hb.times().size(); i++) {
                double t = hb.times().get(i);
                var sim = pred.at(t);

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

}

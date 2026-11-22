package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.util.*;

/**
 * 7-fold cross-validation MAPE using scaled residuals.
 * Mimics Python run_cv.py with 7 folds.
 * Tagged @Tag("slow") — excluded from default mvn test to keep CI fast.
 */
@Tag("slow")
class CrossValidationTest {

    private static final String CSV_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv";

    @Test
    void run7FoldCV() throws Exception {
        var csvReader = new DTBCsvReader(CSV_PATH);
        var all = csvReader.loadAllBatches();
        var train = all.stream().filter(b -> !b.isTest()).toList();

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

        // 7-fold CV by batch
        int nFolds = 7;
        Collections.shuffle(batches, new Random(0));
        int foldSize = batches.size() / nFolds;

        double[] sums = new double[4];
        int[] counts = new int[4];
        String[] labels = {"VCD", "TCD", "Substrate", "Byproduct"};
        long startMs = System.currentTimeMillis();

        for (int f = 0; f < nFolds; f++) {
            int valStart = f * foldSize;
            int valEnd = (f == nFolds - 1) ? batches.size() : (f + 1) * foldSize;
            var valSet = batches.subList(valStart, valEnd);
            var trainSet = new ArrayList<>(batches);
            trainSet.subList(valStart, valEnd).clear();

            System.out.printf("  Fold %d/%d — train: %d, val: %d%n",
                f + 1, nFolds, trainSet.size(), valSet.size());
            Fitter.FitResult fit = fitter.fit(trainSet, 500);
            System.out.printf("    cost=%.4f iter=%d%n", fit.finalCost(), fit.iterations());

            for (var vb : valSet) {
                var pred = fitter.predict(vb, fit.fittedParams());
                for (int i = 1; i < vb.times().size(); i++) {
                    double t = vb.times().get(i);
                    var sim = pred.at(t);
                    double[] obs = {
                        i < vb.observedVCD().size() ? vb.observedVCD().get(i) : Double.NaN,
                        i < vb.observedTCD().size() ? vb.observedTCD().get(i) : Double.NaN,
                        i < vb.observedSubstrate().size() ? vb.observedSubstrate().get(i) : Double.NaN,
                        i < vb.observedByproduct().size() ? vb.observedByproduct().get(i) : Double.NaN
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
        }

        long elapsed = (System.currentTimeMillis() - startMs) / 1000;
        System.out.println("\n  7-FOLD CROSS-VALIDATION MAPE (" + elapsed + "s):");
        double mean = 0; int n = 0;
        for (int j = 0; j < 4; j++) {
            double m = counts[j] > 0 ? sums[j] / counts[j] * 100 : Double.NaN;
            System.out.printf("    %-15s = %7.2f%%  (n=%d)%n", labels[j], m, counts[j]);
            if (!Double.isNaN(m)) { mean += m; n++; }
        }
        System.out.printf("    %-15s = %7.2f%%%n", "MEAN", n > 0 ? mean / n : Double.NaN);
        System.out.println("  Python V3 7-fold CV: VCD=8.6% TCD=8.5% Sub=19.8% Byp=18.5% Mean=13.8%");
    }

}

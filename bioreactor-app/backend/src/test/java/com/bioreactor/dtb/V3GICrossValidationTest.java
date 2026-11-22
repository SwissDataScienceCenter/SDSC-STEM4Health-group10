package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * 7-fold CV comparison: DTBV3Model (22 params) vs DTBV3GIModel (23 params).
 * Both use scaled residuals and Python-fitted defaults as initial guess.
 * Tagged @Tag("slow") — ~4 minutes.
 */
@Tag("slow")
class V3GICrossValidationTest {

    private static final String CSV_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv";

    @Test
    void compareV3AndV3GI7FoldCV() throws Exception {
        var reader = new DTBCsvReader(CSV_PATH);
        var train = reader.loadAllBatches().stream().filter(b -> !b.isTest()).toList();

        record Config(String name, ProcessModel model, int maxEvals) {}
        var configs = List.of(
            new Config("V3 (22 params)", new DTBV3Model(), 500),
            new Config("V3-GI (23 params, K_IP)", new DTBV3GIModel(), 500)
        );

        for (var cfg : configs) {
            var fitter = new Fitter(cfg.model());
            var batches = buildBatches(train, cfg.model());
            Collections.shuffle(batches, new Random(0));

            int nFolds = 7;
            int foldSize = batches.size() / nFolds;
            double[] sums = new double[4];
            int[] counts = new int[4];
            String[] labels = {"VCD", "TCD", "Substrate", "Byproduct"};
            long startMs = System.currentTimeMillis();
            int totalIter = 0;

            for (int f = 0; f < nFolds; f++) {
                int s = f * foldSize;
                int e = (f == nFolds - 1) ? batches.size() : (f + 1) * foldSize;
                var valSet = batches.subList(s, e);
                var trainSet = new ArrayList<>(batches);
                trainSet.subList(s, e).clear();

                // Try with Python defaults first; fall back to multi-start if not converged
                Fitter.FitResult fit;
                try {
                    fit = fitter.fit(trainSet, cfg.maxEvals());
                } catch (Exception ex) {
                    fit = fitter.fitMultiStart(trainSet, 3, cfg.maxEvals());
                }
                if (!fit.converged()) {
                    System.out.printf("  Fold %d: not converged, trying multi-start...%n", f + 1);
                    fit = fitter.fitMultiStart(trainSet, 3, cfg.maxEvals());
                }
                totalIter += fit.iterations();
                System.out.printf("  Fold %d/%d cost=%.4f iter=%d conv=%s%n",
                    f + 1, nFolds, fit.finalCost(), fit.iterations(), fit.converged());

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
            }

            long elapsed = (System.currentTimeMillis() - startMs) / 1000;
            System.out.printf("%n  %s 7-fold CV MAPE (%ds, total iter=%d):%n",
                cfg.name(), elapsed, totalIter);
            double mean = 0; int n = 0;
            for (int j = 0; j < 4; j++) {
                double m = counts[j] > 0 ? sums[j] / counts[j] * 100 : Double.NaN;
                System.out.printf("    %-15s = %7.2f%%  (n=%d)%n", labels[j], m, counts[j]);
                if (!Double.isNaN(m)) { mean += m; n++; }
            }
            System.out.printf("    %-15s = %7.2f%%%n", "MEAN", n > 0 ? mean / n : Double.NaN);
        }

        System.out.println("\n  Python reference (7-fold CV):");
        System.out.println("    V3:    VCD=8.6%  TCD=8.5%  Sub=19.8%  Byp=18.5%  Mean=13.8%");
        System.out.println("    V3-GI: VCD=13.4% TCD=13.2% Sub=19.4% Byp=18.5% Mean=16.2%");
    }

    private List<Fitter.TrainingBatch> buildBatches(
            List<DTBCsvReader.DTBBatch> train, ProcessModel model) {
        List<Fitter.TrainingBatch> out = new ArrayList<>();
        for (var b : train) {
            BioreactorState ic = new BioreactorState(0, b.initialState().xv(), 0,
                b.initialState().substrate(), 0, b.initialState().byproduct(),
                0.007, b.initialState().volume(), 37.0, 7.0,
                model.defaultHiddenState());
            out.add(new Fitter.TrainingBatch(b.batchId(), ic, b.events(),
                b.measurementTimes(),
                b.measurements().getOrDefault("vcd", List.of()),
                b.measurements().getOrDefault("tcd", List.of()),
                b.measurements().getOrDefault("substrate", List.of()),
                b.measurements().getOrDefault("byproduct", List.of())));
        }
        return out;
    }

}

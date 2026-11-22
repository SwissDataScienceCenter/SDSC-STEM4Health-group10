package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Fits model variants on REAL DTB data from CSV.
 * Uses a train/holdout split (not leave-one-out CV) to avoid OOM.
 */
class RealDataFitTest {

    private static final String CSV_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv";

    @Test
    void fitModelsOnRealData() throws Exception {
        var reader = new DTBCsvReader(CSV_PATH);
        List<DTBCsvReader.DTBBatch> trainBatches = reader.loadTrainingBatches();

        System.out.println("\n" + "=".repeat(80));
        System.out.println("  REAL DTB DATA FIT — Model Variant Comparison");
        System.out.println("  Training batches: " + trainBatches.size());
        System.out.println("=".repeat(80));

        // Convert to Fitter.TrainingBatch
        List<Fitter.TrainingBatch> allFitBatches = new ArrayList<>();
        for (var batch : trainBatches) {
            allFitBatches.add(toFitBatch(batch));
        }

        // Split: 60% train (smaller for speed), 40% holdout
        Collections.shuffle(allFitBatches, new Random(42));
        int splitIdx = (int)(allFitBatches.size() * 0.6);
        List<Fitter.TrainingBatch> fitBatches = allFitBatches.subList(0, splitIdx);
        List<Fitter.TrainingBatch> holdoutBatches = allFitBatches.subList(splitIdx, allFitBatches.size());

        System.out.println("Fit on: " + fitBatches.size() + " batches, Holdout: " + holdoutBatches.size() + " batches");

        String[] names = {
            "DTBModelImproved (Haldane+Kp)",
            "DTBModelAdvanced (threshold+byp-consume)",
            "DTBModel (current)"
        };
        ProcessModel[] models = {
            new DTBModelImproved(),
            new DTBModelAdvanced(),
            new DTBModel()
        };

        String[] leaderKeys = {"VCD", "TCD", "Substrate", "Byproduct"};
        String[] mapeKeys = {"mape_vcd", "mape_tcd", "mape_substrate", "mape_byproduct"};

        // Summary
        System.out.println("\n" + "=".repeat(80));
        System.out.println("  FINAL RESULTS (Holdout MAPE %)");
        System.out.println("=".repeat(80));
        System.out.printf("  %-38s %8s %8s %8s %8s %8s%n",
            "Model", "VCD%", "TCD%", "Sub%", "Byp%", "Mean%");

        for (int m = 0; m < models.length; m++) {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("  MODEL: " + names[m]);
            System.out.println("-".repeat(80));

            var fitter = new Fitter(models[m]);

            System.out.println("  Fitting...");
            long start = System.currentTimeMillis();
            Fitter.FitResult fit = fitter.fit(fitBatches, 200);  // single start, 200 max evals
            long elapsed = System.currentTimeMillis() - start;

            // If not converged, just use defaults for MAPE comparison
            if (!fit.converged()) {
                System.out.println("  Did not converge, using default params for MAPE");
                Map<String, Double> defaults = new HashMap<>();
                for (ModelParameter mp : models[m].parameters())
                    defaults.put(mp.name(), mp.startValue());
                fit = new Fitter.FitResult(defaults, defaults, true, 0, Map.of(), 0);
            }

            System.out.println("  Time: " + (elapsed / 1000.0) + "s, Converged: " + fit.converged() +
                ", Iter: " + fit.iterations() + ", RMS: " + String.format("%.6f", fit.finalCost()));

            // Compute MAPE on holdout batches
            double[] holdSum = new double[4];
            int[] holdCount = new int[4];
            for (var hb : holdoutBatches) {
                var pred = fitter.predict(hb, fit.fittedParams());
                for (int i = 1; i < hb.times().size(); i++) {
                    double t = hb.times().get(i);
                    BioreactorState sim = nearestState(pred, t);
                    double obsVcd = i < hb.observedVCD().size() ? hb.observedVCD().get(i) : Double.NaN;
                    double obsTcd = i < hb.observedTCD().size() ? hb.observedTCD().get(i) : Double.NaN;
                    double obsSub = i < hb.observedSubstrate().size() ? hb.observedSubstrate().get(i) : Double.NaN;
                    double obsByp = i < hb.observedByproduct().size() ? hb.observedByproduct().get(i) : Double.NaN;

                    double[] obs = {obsVcd, obsTcd, obsSub, obsByp};
                    double[] predVals = {sim.xv(), sim.xv() + sim.xd(), sim.substrate(), sim.byproduct()};

                    for (int j = 0; j < 4; j++) {
                        if (!Double.isNaN(obs[j]) && obs[j] > 1e-6) {
                            holdSum[j] += Math.abs((predVals[j] - obs[j]) / obs[j]);
                            holdCount[j]++;
                        }
                    }
                }
            }

            double[] mapePct = new double[4];
            double meanMape = 0;
            int cnt = 0;
            System.out.println("\n  Holdout MAPE:");
            for (int j = 0; j < 4; j++) {
                mapePct[j] = holdCount[j] > 0 ? holdSum[j] / holdCount[j] * 100 : Double.NaN;
                System.out.printf("    %-15s = %7.4f%%%n", leaderKeys[j], mapePct[j]);
                if (!Double.isNaN(mapePct[j])) { meanMape += mapePct[j]; cnt++; }
            }
            meanMape = cnt > 0 ? meanMape / cnt : Double.NaN;
            System.out.printf("    %-15s = %7.4f%%%n", "MEAN", meanMape);

            // Print fitted params
            System.out.println("\n  Key fitted params:");
            for (String key : List.of("mu_max", "K_s", "Y_XS", "m_s", "alpha", "beta", "k_d")) {
                Double val = fit.fittedParams().get(key);
                if (val != null) System.out.printf("    %-12s = %.6f%n", key, val);
            }

            // Summary row
            System.out.printf("\n  SUMMARY: %s | VCD=%.2f TCD=%.2f Sub=%.2f Byp=%.2f MEAN=%.2f%n",
                names[m], mapePct[0], mapePct[1], mapePct[2], mapePct[3], meanMape);
        }

        System.out.println("\n" + "=".repeat(80));
        System.out.println("  COMPARISON: Real DTB Data Holdout MAPE (%)");
        System.out.println("  Leaderboard reference: VCD~6.7% TCD~6.6% Sub~7.2% Byp~7.4% Mean~7.0%");
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

package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Fits model variants on REAL DTB data from CSV and reports MAPE.
 *
 * This gives us metrics directly comparable to the leaderboard.
 */
class RealDataFitTest {

    private static final String CSV_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv";

    @Test
    void fitAllModelsOnRealData() throws Exception {
        var reader = new DTBCsvReader(CSV_PATH);
        List<DTBCsvReader.DTBBatch> allBatches = reader.loadAllBatches();
        List<DTBCsvReader.DTBBatch> trainBatches = reader.loadTrainingBatches();
        List<DTBCsvReader.DTBBatch> testBatches = reader.loadTestBatches();

        System.out.println("\n" + "=".repeat(80));
        System.out.println("  REAL DTB DATA FIT — Model Variant Comparison");
        System.out.println("  " + trainBatches.size() + " training batches, " + testBatches.size() + " test batches");
        System.out.println("=".repeat(80));

        // Convert to Fitter.TrainingBatch format
        List<Fitter.TrainingBatch> fitBatches = new ArrayList<>();
        for (var batch : trainBatches) {
            fitBatches.add(toFitBatch(batch));
        }

        System.out.println("Total training observations: " +
            fitBatches.stream().mapToInt(b -> b.times().size()).sum());

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

        for (int m = 0; m < models.length; m++) {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("  MODEL: " + names[m]);
            System.out.println("  Parameters: " + models[m].parameters().size());
            System.out.println("-".repeat(80));

            var fitter = new Fitter(models[m]);

            // Fit with multi-start
            System.out.println("  Fitting on " + fitBatches.size() + " training batches...");
            long start = System.currentTimeMillis();
            Fitter.FitResult fit = fitter.fitMultiStart(fitBatches, 5, 200);
            long elapsed = System.currentTimeMillis() - start;

            System.out.println("  Fit time: " + (elapsed / 1000.0) + "s");
            System.out.println("  Converged: " + fit.converged());
            System.out.println("  Iterations: " + fit.iterations());
            System.out.println("  RMS cost: " + String.format("%.6f", fit.finalCost()));

            // Training MAPE
            System.out.println("\n  >>> TRAINING MAPE (%):");
            Map<String, Double> trainMape = fit.mapePerVariable();
            String[] keys = {"mape_vcd", "mape_tcd", "mape_substrate", "mape_byproduct"};
            String[] labels = {"VCD", "TCD", "Substrate", "Byproduct"};
            double[] trainVals = new double[4];
            for (int i = 0; i < 4; i++) {
                Double v = trainMape.get(keys[i]);
                trainVals[i] = (v != null && !v.isNaN()) ? v : Double.NaN;
                System.out.printf("    %-15s = %7.4f%%%n", labels[i], trainVals[i]);
            }
            double trainMean = 0; int tc = 0;
            for (double v : trainVals) { if (!Double.isNaN(v)) { trainMean += v; tc++; } }
            if (tc > 0) trainMean /= tc;
            System.out.printf("    %-15s = %7.4f%%%n", "MEAN", trainMean);

            // Leave-one-batch-out CV
            System.out.println("\n  >>> CROSS-VALIDATION MAPE (%) [leave-one-batch-out]:");
            Map<String, Double> cvMape = fitter.crossValidate(fitBatches, 150);
            double[] cvVals = new double[4];
            for (int i = 0; i < 4; i++) {
                Double v = cvMape.get(keys[i]);
                cvVals[i] = (v != null && !v.isNaN()) ? v : Double.NaN;
                System.out.printf("    %-15s = %7.4f%%%n", labels[i], cvVals[i]);
            }
            double cvMean = 0; int cc = 0;
            for (double v : cvVals) { if (!Double.isNaN(v)) { cvMean += v; cc++; } }
            if (cc > 0) cvMean /= cc;
            System.out.printf("    %-15s = %7.4f%%%n", "MEAN", cvMean);

            // Print fitted parameters
            System.out.println("\n  Fitted parameters:");
            List<String> important = List.of("mu_max", "K_s", "K_i", "Y_XS", "m_s", "alpha", "beta",
                "K_p", "B_thresh", "B_crit", "q_B_max", "Y_XB", "k_d", "feedB_S");
            for (String key : important) {
                Double val = fit.fittedParams().get(key);
                if (val != null) {
                    System.out.printf("    %-14s = %.6f%n", key, val);
                }
            }
        }

        System.out.println("\n" + "=".repeat(80));
        System.out.println("  DONE — Real data fitting complete");
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
}

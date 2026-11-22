package com.bioreactor.dtb;

import com.bioreactor.engine.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Generates a DTB challenge submission CSV without requiring a database.
 *
 * Usage:
 *   java -cp target/classes:target/dependency/* com.bioreactor.dtb.DTBSubmissionGenerator \
 *     /path/to/bioreaction_remaining.csv
 *     /path/to/bioreaction_template.csv
 *     /path/to/output_submission.csv
 *     [--skip-cv] [--no-fit] [--model=V3|V3Constrained]
 *
 * By default fits DTBV3Model (22 params, Python defaults) on all training batches.
 * --skip-cv: skip the cross-validation step (faster)
 * --no-fit:  use default parameters without fitting
 */
public class DTBSubmissionGenerator {

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println("Usage: DTBSubmissionGenerator <input-csv> <template-csv> <output-csv>");
            System.out.println("  input-csv:    path to bioreaction_remaining.csv");
            System.out.println("  template-csv: path to bioreaction_template.csv (156 rows)");
            System.out.println("  output-csv:   path for submission CSV");
            System.exit(1);
        }

        String inputPath = args[0];
        String templatePath = args[1];
        String outputPath = args[2];

        boolean skipCV = false;
        boolean noFit = false;
        String modelSel = "V3";  // V3 or V3Constrained
        for (String a : args) {
            if ("--skip-cv".equals(a)) skipCV = true;
            if ("--no-fit".equals(a)) noFit = true;
            if (a.startsWith("--model=")) modelSel = a.substring("--model=".length());
        }

        System.out.println("=".repeat(70));
        System.out.println("  DTB SUBMISSION GENERATOR");
        System.out.println("  Model: DTBV3ConstrainedModel (12 free params)");
        System.out.println("  Input:    " + inputPath);
        System.out.println("  Template: " + templatePath);
        System.out.println("  Output:   " + outputPath);
        System.out.println("=".repeat(70));

        // Load data
        var csvReader = new DTBCsvReader(inputPath);
        List<DTBCsvReader.DTBBatch> all = csvReader.loadAllBatches();
        List<DTBCsvReader.DTBBatch> train = all.stream().filter(b -> !b.isTest()).toList();
        List<DTBCsvReader.DTBBatch> test = all.stream().filter(DTBCsvReader.DTBBatch::isTest).toList();

        System.out.println("\nLoaded " + all.size() + " batches: " +
            train.size() + " train, " + test.size() + " test");

        ProcessModel model = modelSel.equals("V3Constrained") ?
            new DTBV3ConstrainedModel() : new DTBV3Model();
        var fitter = new Fitter(model);
        System.out.println("  Model: " + model.getClass().getSimpleName() +
            " (" + model.parameters().size() + " params)");

        // Convert training batches
        List<Fitter.TrainingBatch> fitBatches = new ArrayList<>();
        for (var b : train) {
            BioreactorState ic = new BioreactorState(0, b.initialState().xv(), 0,
                b.initialState().substrate(), 0, b.initialState().byproduct(),
                0.007, b.initialState().volume(), 37.0, 7.0, new double[]{1.0, 0.0});
            fitBatches.add(new Fitter.TrainingBatch(b.batchId(), ic, b.events(),
                b.measurementTimes(),
                b.measurements().getOrDefault("vcd", List.of()),
                b.measurements().getOrDefault("tcd", List.of()),
                b.measurements().getOrDefault("substrate", List.of()),
                b.measurements().getOrDefault("byproduct", List.of())));
        }

        // Optional: compute training MAPE via CV
        if (!skipCV) {
            System.out.println("\nComputing training MAPE (3-fold CV on 15 batches)...");
            Collections.shuffle(fitBatches, new Random(42));
            List<Fitter.TrainingBatch> cvBatches = fitBatches.subList(0, Math.min(15, fitBatches.size()));
            Map<String, Double> cvMape = fitter.crossValidate(cvBatches, 300);
            System.out.printf("  VCD=%.1f%% TCD=%.1f%% Sub=%.1f%% Byp=%.1f%%%n",
                cvMape.getOrDefault("mape_vcd", Double.NaN),
                cvMape.getOrDefault("mape_tcd", Double.NaN),
                cvMape.getOrDefault("mape_substrate", Double.NaN),
                cvMape.getOrDefault("mape_byproduct", Double.NaN));
        }

        // Fit or use defaults
        Fitter.FitResult fit;
        if (noFit) {
            System.out.println("\nSkipping fit (--no-fit), using default parameters");
            Map<String, Double> defaults = new HashMap<>();
            for (ModelParameter mp : model.parameters()) defaults.put(mp.name(), mp.startValue());
            fit = new Fitter.FitResult(defaults, defaults, true, 0.0, Map.of(), 0);
        } else {
            System.out.println("\nFitting on " + fitBatches.size() + " training batches...");
            long startMs = System.currentTimeMillis();
            fit = fitter.fit(fitBatches, 1000);
            long elapsed = (System.currentTimeMillis() - startMs) / 1000;

            if (!fit.converged()) {
                System.out.println("WARNING: Fitter did not converge. Using default parameters.");
                Map<String, Double> defaults = new HashMap<>();
                for (ModelParameter mp : model.parameters()) defaults.put(mp.name(), mp.startValue());
                fit = new Fitter.FitResult(defaults, defaults, false, Double.NaN,
                    Map.of(), 0);
            }

            System.out.println("Time: " + elapsed + "s | Converged: " + fit.converged() +
                " | Iter: " + fit.iterations() + " | RMS: " +
                String.format("%.6f", fit.converged() ? fit.finalCost() : Double.NaN));
        }

        System.out.println("\nFitted parameters:");
        for (var mp : model.parameters()) {
            Double val = fit.fittedParams().get(mp.name());
            if (val != null) {
                System.out.printf("  %-15s = %12.6f  (start: %12.6f)%n",
                    mp.name(), val, mp.startValue());
            }
        }

        // Read the submission template to get the exact time points per batch
        System.out.println("\nReading submission template...");
        Map<String, List<Double>> templateTimes = new LinkedHashMap<>();
        int templateRows = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(templatePath))) {
            String header = br.readLine(); // skip header
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",", -1);
                if (parts.length < 2) continue;
                String bid = parts[0].trim();
                double t = Double.parseDouble(parts[1].trim());
                templateTimes.computeIfAbsent(bid, k -> new ArrayList<>()).add(t);
                templateRows++;
            }
        }
        System.out.println("Template has " + templateRows + " rows across " +
            templateTimes.size() + " batches");

        // Build an index: test batch by ID
        Map<String, DTBCsvReader.DTBBatch> testById = new LinkedHashMap<>();
        for (var tb : test) testById.put(tb.batchId(), tb);

        // Generate predictions for each template time point
        System.out.println("\nGenerating predictions...");
        int totalRows = 0;
        Map<String, RK4Simulator.SimResult> predictionCache = new HashMap<>();

        try (PrintWriter w = new PrintWriter(new FileWriter(outputPath, StandardCharsets.UTF_8))) {
            w.println("batch_id,elapsed_time,vcd,tcd,substrate,byproduct");

            for (var entry : templateTimes.entrySet()) {
                String batchId = entry.getKey();
                List<Double> times = entry.getValue();
                DTBCsvReader.DTBBatch tb = testById.get(batchId);

                if (tb == null) {
                    System.out.println("  WARNING: " + batchId + " not found in data, writing zeros");
                    for (double t : times) {
                        w.printf("%s,%.6f,%.6f,%.6f,%.6f,%.6f%n",
                            batchId, t, 0.0, 0.0, 0.0, 0.0);
                        totalRows++;
                    }
                    continue;
                }

                // Simulate once per test batch
                RK4Simulator.SimResult pred = predictionCache.get(batchId);
                if (pred == null) {
                    BioreactorState ic = new BioreactorState(0, tb.initialState().xv(), 0,
                        tb.initialState().substrate(), 0, tb.initialState().byproduct(),
                        0.007, tb.initialState().volume(), 37.0, 7.0,
                        new double[]{1.0, 0.0});
                    Fitter.TrainingBatch fb = new Fitter.TrainingBatch(tb.batchId(), ic,
                        tb.events(), tb.measurementTimes(),
                        tb.measurements().getOrDefault("vcd", List.of()),
                        tb.measurements().getOrDefault("tcd", List.of()),
                        tb.measurements().getOrDefault("substrate", List.of()),
                        tb.measurements().getOrDefault("byproduct", List.of()));
                    pred = fitter.predict(fb, fit.fittedParams());
                    predictionCache.put(batchId, pred);
                }

                // Interpolate at each template time
                for (double t : times) {
                    var s = nearestState(pred, t);
                    double vcd = Math.max(0, s.xv());
                    double tcd = Math.max(0, s.xv() + s.xd());
                    double sub = Math.max(0, s.substrate());
                    double byp = Math.max(0, s.byproduct());

                    w.printf("%s,%.6f,%.6f,%.6f,%.6f,%.6f%n",
                        batchId, t, vcd, tcd, sub, byp);
                    totalRows++;
                }
            }
        }

        System.out.println("Done! Wrote " + totalRows + " rows to " + outputPath);
        System.out.println("Expected: " + templateRows + " template rows");
        System.out.println("=".repeat(70));
    }

    private static BioreactorState nearestState(RK4Simulator.SimResult r, double t) {
        int idx = 0;
        for (int i = 0; i < r.times().size(); i++) {
            if (r.times().get(i) >= t) { idx = i; break; }
            idx = i;
        }
        if (idx >= r.states().size()) return r.states().getLast();
        return r.states().get(idx);
    }
}

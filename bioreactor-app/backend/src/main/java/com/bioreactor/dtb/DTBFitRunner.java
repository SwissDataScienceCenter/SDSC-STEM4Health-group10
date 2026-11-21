package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import javax.sql.DataSource;
import java.sql.SQLException;
import java.util.*;

/**
 * Fits the DTB model on real training data, reports MAPE,
 * and generates prediction CSV. Run via: `mvn spring-boot:run -Dspring-boot.run.profiles=fit`
 */
@Component
@org.springframework.boot.autoconfigure.condition.ConditionalOnProperty(name = "bioreactor.fit.enabled", havingValue = "true")
public class DTBFitRunner implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(DTBFitRunner.class);
    private final DTBDataReader reader;
    private final ProcessModel model;
    private final Fitter fitter;

    public DTBFitRunner(DataSource dataSource) {
        this.reader = new DTBDataReader(dataSource);
        this.model = new DTBModel();
        this.fitter = new Fitter(model);
    }

    @Override
    public void run(String... args) throws SQLException {
        log.info("Loading training batches...");
        List<DTBDataReader.DTBBatch> trainBatches = reader.loadTrainingBatches();
        log.info("Loaded {} training batches", trainBatches.size());

        // Convert to fitter batches
        List<Fitter.TrainingBatch> fitBatches = trainBatches.stream()
            .map(this::toFitBatch)
            .toList();

        log.info("Fitting...");
        Fitter.FitResult fit = fitter.fit(fitBatches, 200);
        log.info("Fit complete. Converged={}", fit.converged());
        log.info("Fitted params: {}", fit.fittedParams());
        log.info("MAPE: {}", fit.mapePerVariable());

        // Leave-one-out CV
        log.info("Cross-validation (leave-one-batch-out)...");
        Map<String, Double> cvMape = fitter.crossValidate(fitBatches, 100);
        log.info("CV MAPE: {}", cvMape);

        // Generate predictions for test batches
        log.info("Loading test batches...");
        List<DTBDataReader.DTBBatch> testBatches = reader.loadTestBatches();
        log.info("Loaded {} test batches", testBatches.size());

        for (var batch : testBatches) {
            var fb = toFitBatch(batch);
            var sim = fitter.predict(fb, fit.fittedParams());
            log.info("Prediction for {}: {} states, {} times",
                batch.batchId(), sim.states().size(), sim.times().size());
        }

        // Print summary
        System.out.println("\n===== FITTING RESULTS =====");
        System.out.println("Converged: " + fit.converged());
        System.out.println("Iterations: " + fit.iterations());
        System.out.println("Fitted parameters:");
        for (var entry : fit.fittedParams().entrySet()) {
            System.out.printf("  %-15s = %.6f%n", entry.getKey(), entry.getValue());
        }
        System.out.println("\nTraining MAPE:");
        for (var entry : fit.mapePerVariable().entrySet()) {
            System.out.printf("  %-15s = %.4f%%%n", entry.getKey(), entry.getValue() * 100);
        }
        System.out.println("\nCV MAPE:");
        for (var entry : cvMape.entrySet()) {
            System.out.printf("  %-15s = %.4f%%%n", entry.getKey(), entry.getValue() * 100);
        }
        System.out.println("===========================");
    }

    private Fitter.TrainingBatch toFitBatch(DTBDataReader.DTBBatch b) {
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

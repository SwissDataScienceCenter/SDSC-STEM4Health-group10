package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Diagnostic: simulates first 3 training batches with V3 Python defaults
 * and dumps the trajectory at measurement times.
 */
class SingleBatchDiagnosticTest {

    private static final String CSV_PATH = "/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv";

    @Test
    void dumpBatchB1Trajectory() throws Exception {
        var csvReader = new DTBCsvReader(CSV_PATH);
        var batches = csvReader.loadAllBatches();
        var train = batches.stream().filter(b -> !b.isTest()).toList();

        var model = new DTBV3Model();
        var fitter = new Fitter(model);
        Map<String, Double> params = model.defaultParams();

        System.out.println("Training batches: " + train.size());
        for (var b : train) System.out.println("  " + b.batchId() + " times=" + b.measurementTimes().size() + " events=" + b.events().size());

        System.out.printf("\n%-10s %8s %10s %10s %10s %10s %10s%n",
            "t", "batch", "VCD", "TCD", "Sub", "Byp", "Vol");
        System.out.println("-".repeat(70));

        for (var b : train.stream().limit(3).toList()) {
            var tb = new Fitter.TrainingBatch(b.batchId(), b.initialState(), b.events(),
                b.measurementTimes(),
                b.measurements().getOrDefault("vcd", List.of()),
                b.measurements().getOrDefault("tcd", List.of()),
                b.measurements().getOrDefault("substrate", List.of()),
                b.measurements().getOrDefault("byproduct", List.of()));
            var pred = fitter.predict(tb, params);
            for (int i = 0; i < b.measurementTimes().size(); i++) {
                double t = b.measurementTimes().get(i);
                var sim = pred.at(t);
                double vcd = sim.xv();
                double tcd = sim.xv() + sim.xd();
                double sub = Math.max(0, sim.substrate());
                double byp = Math.max(0, sim.byproduct());
                double vol = sim.volume();

                System.out.printf("%10.2f %-10s %10.4f %10.4f %10.4f %10.4f %10.4f%n",
                    t, b.batchId(), vcd, tcd, sub, byp, vol);
            }
        }

        // Also print feed events for the first batch
        var b0 = train.get(0);
        System.out.println("\n--- Events for " + b0.batchId() + " ---");
        for (var ev : b0.events()) {
            System.out.printf("  t=%.4f  %s  dV=%.4f  name=%s%n",
                ev.time(), ev.type(), ev.dV(), ev.feedName());
        }
    }

}

package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import java.sql.*;
import javax.sql.DataSource;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import java.util.*;
import java.io.*;

/**
 * Standalone fitter for DTB data. Runs from CLI without Spring.
 *
 * Usage: java -cp target/classes:target/dependency/* com.bioreactor.dtb.DTBCliFitter
 */
public class DTBCliFitter {

    public static void main(String[] args) throws Exception {
        // Bootstrap HikariCP manually
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:postgresql://localhost:5432/bioreactor");
        config.setUsername("appuser");
        config.setPassword("appuser");
        config.setMaximumPoolSize(2);
        HikariDataSource ds = new HikariDataSource(config);

        DTBDataReader reader = new DTBDataReader(ds);
        DTBV2Model model = new DTBV2Model();
        Fitter fitter = new Fitter(model);

        System.out.println("Loading training batches...");
        List<DTBDataReader.DTBBatch> trainBatches = reader.loadTrainingBatches();
        System.out.println("Loaded " + trainBatches.size() + " training batches");

        // Convert to fitter batches
        List<Fitter.TrainingBatch> fitBatches = new ArrayList<>();
        for (var b : trainBatches) {
            var vcd = b.measurements().getOrDefault("vcd", List.of());
            var tcd = b.measurements().getOrDefault("tcd", List.of());
            var sub = b.measurements().getOrDefault("substrate", List.of());
            var byp = b.measurements().getOrDefault("byproduct", List.of());
            fitBatches.add(new Fitter.TrainingBatch(
                b.batchId(), b.initialState(), b.events(),
                b.measurementTimes(), vcd, tcd, sub, byp
            ));
        }

        // Check batch sizes
        int totalObs = 0;
        for (var b : fitBatches) {
            totalObs += b.times().size();
        }
        System.out.println("Total observation points: " + totalObs);

        System.out.println("Fitting...");
        long start = System.currentTimeMillis();

        // Quick test: simulate first batch with default params and check
        System.out.println("\n--- Quick check: Batch1 with default params ---");
        var b1 = fitBatches.get(0);
        Map<String, Double> defaultParams = new HashMap<>();
        for (var mp : model.parameters()) {
            defaultParams.put(mp.name(), mp.startValue());
        }
        System.out.println("Default params: " + defaultParams);

        var pred = fitter.predict(b1, defaultParams);
        System.out.println("Simulated states count: " + pred.states().size());
        System.out.println("First few states:");
        for (int i = 0; i < Math.min(5, pred.states().size()); i++) {
            var s = pred.states().get(i);
            System.out.printf("  t=%.4f: Xv=%.4f, Xd=%.4f, S=%.4f, B=%.4f, V=%.4f%n",
                pred.times().get(i), s.xv(), s.xd(), s.substrate(), s.byproduct(), s.volume());
        }

        // DEBUG: show events for Batch1
        System.out.println("\n--- Events for Batch1 ---");
        for (var ev : b1.events()) {
            System.out.printf("  t=%.6f type=%s dV=%.6f V_meas=%s conc=%s%n",
                ev.time(), ev.type(), ev.dV(), ev.measuredVolume(), ev.feedConcentrations());
        }

        // Check observed vs predicted for Batch1
        System.out.println("\nObserved vs Predicted (Batch1, default params):");
        for (int i = 0; i < Math.min(b1.times().size(), 10); i++) {
            double t = b1.times().get(i);
            BioreactorState sim = nearestState(pred, t);
            double obsVcd = i < b1.observedVCD().size() ? b1.observedVCD().get(i) : Double.NaN;
            double obsSub = i < b1.observedSubstrate().size() ? b1.observedSubstrate().get(i) : Double.NaN;
            double obsByp = i < b1.observedByproduct().size() ? b1.observedByproduct().get(i) : Double.NaN;
            System.out.printf("  t=%.3f: pred(vcd=%.2f,sub=%.2f,byp=%.2f) obs(vcd=%.2f,sub=%.2f,byp=%.2f)%n",
                t, sim.xv(), sim.substrate(), sim.byproduct(), obsVcd, obsSub, obsByp);
        }

        // Quick MAPE computation at default params
        System.out.println("\n--- MAPE at default params ---");
        var mapeDefault = computeMAPE(fitBatches, defaultParams, fitter);
        System.out.println("MAPE (default): " + mapeDefault);

        long elapsed = System.currentTimeMillis() - start;
        System.out.println("\nElapsed: " + (elapsed / 1000.0) + " s");

        ds.close();
    }

    private static BioreactorState nearestState(Simulator.SimulationResult r, double t) {
        int idx = 0;
        var ts = r.times();
        for (int i = 0; i < ts.size(); i++) {
            if (ts.get(i) >= t) { idx = i; break; }
            idx = i;
        }
        return r.states().get(idx);
    }

    private static Map<String, Double> computeMAPE(List<Fitter.TrainingBatch> batches,
                                                     Map<String, Double> params, Fitter fitter) {
        double sumVcd = 0, sumTcd = 0, sumSub = 0, sumByp = 0;
        int nVcd = 0, nTcd = 0, nSub = 0, nByp = 0;
        for (var b : batches) {
            var pred = fitter.predict(b, params);
            for (int i = 1; i < b.times().size(); i++) {
                double t = b.times().get(i);
                var sim = nearestState(pred, t);
                double obsVcd = i < b.observedVCD().size() ? b.observedVCD().get(i) : Double.NaN;
                double obsTcd = i < b.observedTCD().size() ? b.observedTCD().get(i) : Double.NaN;
                double obsSub = i < b.observedSubstrate().size() ? b.observedSubstrate().get(i) : Double.NaN;
                double obsByp = i < b.observedByproduct().size() ? b.observedByproduct().get(i) : Double.NaN;
                if (!Double.isNaN(obsVcd) && obsVcd > 1e-6) {
                    sumVcd += Math.abs((sim.xv() - obsVcd) / obsVcd); nVcd++;
                }
                if (!Double.isNaN(obsTcd) && obsTcd > 1e-6) {
                    sumTcd += Math.abs(((sim.xv()+sim.xd()) - obsTcd) / obsTcd); nTcd++;
                }
                if (!Double.isNaN(obsSub) && obsSub > 1e-6) {
                    sumSub += Math.abs((sim.substrate() - obsSub) / obsSub); nSub++;
                }
                if (!Double.isNaN(obsByp) && obsByp > 1e-6) {
                    sumByp += Math.abs((sim.byproduct() - obsByp) / obsByp); nByp++;
                }
            }
        }
        return Map.of("mape_vcd", nVcd > 0 ? sumVcd/nVcd*100 : Double.NaN,
                       "mape_tcd", nTcd > 0 ? sumTcd/nTcd*100 : Double.NaN,
                       "mape_substrate", nSub > 0 ? sumSub/nSub*100 : Double.NaN,
                       "mape_byproduct", nByp > 0 ? sumByp/nByp*100 : Double.NaN);
    }
}

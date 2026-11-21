package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import java.util.*;

/**
 * Quick parameter sweep to find better substrate MAPE.
 */
public class DTBSweep {

    public static void main(String[] args) throws Exception {
        HikariConfig cfg = new HikariConfig();
        cfg.setJdbcUrl("jdbc:postgresql://localhost:5432/bioreactor");
        cfg.setUsername("appuser");
        cfg.setPassword("appuser");
        HikariDataSource ds = new HikariDataSource(cfg);

        DTBDataReader reader = new DTBDataReader(ds);
        System.out.println("Loading training batches...");
        List<DTBDataReader.DTBBatch> batches = reader.loadTrainingBatches();
        System.out.println("Loaded " + batches.size() + " training batches");

        DTBV2Model model = new DTBV2Model();
        Map<String, Double> base = model.defaultParams();
        RK4Simulator rk4 = new RK4Simulator(model, 0.01);

        // Baseline
        System.out.println("\n=== Baseline (V2 defaults) ===");
        System.out.println(DTBCliFitter.computeMAPE(batches, base, rk4, model));

        // Try V3-like params: higher mu_max, higher Ks, much higher Yxs, lower m
        Map<String, Double> p1 = new HashMap<>(base);
        p1.put("mu_max", 5.0);
        p1.put("Ks", 5.0);
        p1.put("Yxs", 200.0);
        p1.put("m", 0.1);
        p1.put("C_b", 250.0);  // try higher feed concentration
        System.out.println("\n=== V3-like (mu_max=5, Ks=5, Yxs=200, C_b=250) ===");
        System.out.println(DTBCliFitter.computeMAPE(batches, p1, rk4, model));

        // Try moderate Yxs
        Map<String, Double> p2 = new HashMap<>(base);
        p2.put("Yxs", 50.0);
        p2.put("C_b", 150.0);
        System.out.println("\n=== Moderate (Yxs=50, C_b=150) ===");
        System.out.println(DTBCliFitter.computeMAPE(batches, p2, rk4, model));

        // Try very low consumption
        Map<String, Double> p3 = new HashMap<>(base);
        p3.put("mu_max", 2.0);
        p3.put("Yxs", 100.0);
        p3.put("m", 0.02);
        p3.put("C_b", 500.0);
        System.out.println("\n=== Low consumption (mu_max=2, Yxs=100, m=0.02, C_b=500) ===");
        System.out.println(DTBCliFitter.computeMAPE(batches, p3, rk4, model));

        // Try the ridge regression's idea: substrate fed by feed_substrate columns have high C
        Map<String, Double> p4 = new HashMap<>(base);
        p4.put("C_sub", 500.0); // higher substrate feed
        p4.put("C_b", 200.0);
        p4.put("Yxs", 80.0);
        System.out.println("\n=== High feeds (C_sub=500, C_b=200, Yxs=80) ===");
        System.out.println(DTBCliFitter.computeMAPE(batches, p4, rk4, model));

        ds.close();
    }
}

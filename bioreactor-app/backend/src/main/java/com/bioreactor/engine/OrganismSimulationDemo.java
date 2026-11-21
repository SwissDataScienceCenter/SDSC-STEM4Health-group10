package com.bioreactor.engine;

import java.util.*;

/**
 * Manual test: load E. coli K-12 from DB and simulate a batch.
 * Run via: java -cp ... com.bioreactor.engine.OrganismSimulationDemo
 */
public class OrganismSimulationDemo {

    public static void main(String[] args) {
        // Build config manually (simulating DB load)
        var substrates = List.of(
            new ConfigurableModel.SubstrateConfig("glucose", "glucose", 1.5, 0.01, 1),
            new ConfigurableModel.SubstrateConfig("glycerol", "glycerol", 0.8, 0.05, 2)
        );
        var products = List.<ConfigurableModel.ProductConfig>of();
        var byproducts = List.of(
            new ConfigurableModel.ProductConfig("acetate", "byproduct", 0.0, 0.02, 0.15, 0.80, "linear", 15.0)
        );

        var config = new ConfigurableModel.OrganismConfig(
            "ecoli_k12", "E. coli K-12 MG1655",
            "andrews", 0.70, 0.02, 50.0, 0.50, 0.025, null, 1.0,
            0.010, 0.005,
            substrates, products, byproducts,
            8.0, 37.0, 45.0, 5.0, 7.0, 9.0, 0.8,
            0.40, 0.95, 0.01, 19.0, 14.0);

        ConfigurableModel model = new ConfigurableModel(config);
        System.out.println("Loaded: " + config.name());
        System.out.println("Hidden states: " + model.hiddenStateCount());
        System.out.println("Parameters: " + model.parameters().size());

        // Initial state using standard BioreactorState layout
        BioreactorState init = new BioreactorState(
            0.0,    // time
            0.1,    // Xv = 0.1 g/L
            0.0,    // Xd
            10.0,   // S (glucose) = 10 g/L
            0.0,    // P (product)
            0.0,    // B (acetate byproduct)
            0.007,  // DO
            1.0,    // V
            37.0,   // T
            7.0);   // pH

        Map<String, Double> params = new HashMap<>();
        for (ModelParameter mp : model.parameters()) params.put(mp.name(), mp.startValue());

        RK4Simulator sim = new RK4Simulator(model, 0.01);
        long t0 = System.nanoTime();
        var result = sim.simulate(init, params, List.of(), 24.0);
        long elapsed = System.nanoTime() - t0;

        System.out.println("\nSimulation complete in " + (elapsed / 1e6) + " ms");
        System.out.println("Output points: " + result.times().size());

        // Show key time points
        double[] checkTimes = {0.0, 2.0, 6.0, 12.0, 18.0, 24.0};
        System.out.printf("%-8s %10s %10s %10s %10s %10s%n", "Time", "Xv", "Xd", "Glucose", "Acetate", "DO");
        for (double t : checkTimes) {
            var s = sim.stateAt(result.times(), result.states(), t);
            System.out.printf("%-8.1f %10.4f %10.4f %10.4f %10.4f %10.4f%n",
                t, s.xv(), s.xd(), s.substrate(), s.byproduct(), s.dissolvedO2());
        }

        // Benchmark: simulate all 9 organisms
        System.out.println("\n=== Benchmark: 9 organisms, 24h each ===");
        String[] orgs = {"ecoli_k12", "ecoli_bl21", "scerevisiae", "bsubtilis", "ppastoris",
                         "cglutamicum", "aniger", "lactobacillus", "zmobilis"};
        t0 = System.nanoTime();
        for (String orgId : orgs) {
            var cfg = configFor(orgId);
            var m = new ConfigurableModel(cfg);
            var s = new RK4Simulator(m, 0.01);
            var p = new HashMap<String, Double>();
            for (ModelParameter mp : m.parameters()) p.put(mp.name(), mp.startValue());
            int dim2 = BioreactorState.CORE_STATE_SIZE + m.hiddenStateCount();
            BioreactorState si = new BioreactorState(
                0.0, 0.5, 0.0, 20.0, 0.0, 0.0, 0.007, 1.0, cfg.tOpt(), cfg.pHOpt());
            s.simulate(si, p, List.of(), 24.0);
        }
        elapsed = System.nanoTime() - t0;
        System.out.println("Total: " + (elapsed / 1e6) + " ms (" + (elapsed / 1e6 / 9) + " ms/org)");
    }

    private static ConfigurableModel.OrganismConfig configFor(String id) {
        // Simplified configs for benchmarking
        var sub = List.of(new ConfigurableModel.SubstrateConfig("glucose", "glucose", 1.0, 0.05, 1));
        return new ConfigurableModel.OrganismConfig(id, id, "monod", 0.5, 0.05, null, 0.45, 0.02,
            null, null, 0.01, 0.005, sub, List.of(), List.of(),
            15.0, 37.0, 45.0, 5.0, 7.0, 9.0, 0.8,
            0.3, 0.8, 0.01, 18.0, 14.0);
    }
}

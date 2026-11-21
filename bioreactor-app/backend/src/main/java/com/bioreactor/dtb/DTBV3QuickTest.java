package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import java.util.*;

/**
 * Quick MAPE measurement for V3 model on real data.
 */
public class DTBV3QuickTest {

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

        DTBV3Model model = new DTBV3Model();
        Map<String, Double> params = model.defaultParams();
        RK4Simulator rk4 = new RK4Simulator(model, 0.01);

        System.out.println("V3 model: hiddenStateCount=" + model.hiddenStateCount());
        System.out.println("Default params: " + params);

        long start = System.currentTimeMillis();
        Map<String, Double> mape = computeMAPE(batches, params, rk4, model);
        long elapsed = System.currentTimeMillis() - start;

        System.out.println("\n--- V3 MAPE (default params, RK4 dt=0.01) ---");
        for (var e : mape.entrySet()) {
            System.out.printf("  %s = %.4f%%%n", e.getKey(), e.getValue());
        }
        System.out.println("Elapsed: " + (elapsed / 1000.0) + " s");

        // Also try with fitted params from Victor's terminal
        Map<String, Double> fitted = Map.ofEntries(
            Map.entry("mu_max", 4.71), Map.entry("Ks", 2.71), Map.entry("K_N", 0.353),
            Map.entry("Yxn", 42.0), Map.entry("C_Nb", 0.0001),
            Map.entry("kd0", 0.000001), Map.entry("kd_N", 0.0317), Map.entry("K_dN", 0.00036),
            Map.entry("kd_D", 0.00437), Map.entry("k_r", 1.09),
            Map.entry("k_lys", 0.0491),
            Map.entry("Yxs", 9999.0), Map.entry("m", 0.0580),
            Map.entry("q_ovf", 1.93), Map.entry("K_ovf", 2.61),
            Map.entry("n_ovf", 20.0), Map.entry("K_iP", 0.176),
            Map.entry("Yps", 1.34), Map.entry("qPc", 2.0), Map.entry("K_Pc", 28.4),
            Map.entry("C_sub", 1.0), Map.entry("C_b", 74.5), Map.entry("K_IP", 0.76)
        );
        // Fill missing params from defaults
        Map<String, Double> p2 = new HashMap<>(params);
        p2.putAll(fitted);

        System.out.println("\n--- V3 MAPE (Victor's fitted params) ---");
        long start2 = System.currentTimeMillis();
        Map<String, Double> mape2 = computeMAPE(batches, p2, rk4, model);
        long elapsed2 = System.currentTimeMillis() - start2;
        for (var e : mape2.entrySet()) {
            System.out.printf("  %s = %.4f%%%n", e.getKey(), e.getValue());
        }
        System.out.println("Elapsed: " + (elapsed2 / 1000.0) + " s");

        ds.close();
    }

    public static Map<String, Double> computeMAPE(
        List<DTBDataReader.DTBBatch> batches,
        Map<String, Double> params,
        RK4Simulator rk4,
        ProcessModel model
    ) {
        double sumVCD = 0, sumTCD = 0, sumSub = 0, sumByp = 0;
        int countVCD = 0, countTCD = 0, countSub = 0, countByp = 0;

        for (var batch : batches) {
            // Create initial state with hidden states
            double[] hidden = new double[model.hiddenStateCount()];
            // N(0) = C_Nb (normalized to match Python initial condition)
            // Actually from the Python: N0 = 1.0 for first batch
            if (model.hiddenStateCount() > 0) hidden[0] = 1.0;  // N0 = 1
            // D0 = 0
            if (model.hiddenStateCount() > 1) hidden[1] = 0.0;

            BioreactorState init = new BioreactorState(
                0.0, batch.initialState().xv(), batch.initialState().xd(),
                batch.initialState().substrate(), 0.0, 0.0,
                batch.initialState().dissolvedO2(), batch.initialState().volume(),
                batch.initialState().temperature(), batch.initialState().pH(),
                hidden);

            double tEnd = batch.events().stream()
                .mapToDouble(Simulator.Event::time).max().orElse(14.0);
            var sim = rk4.simulate(init, params, batch.events(), tEnd);

            // Evaluate at measurement times
            for (int i = 0; i < batch.measurementTimes().size(); i++) {
                double t = batch.measurementTimes().get(i);
                var s = rk4.stateAt(sim.times(), sim.states(), t);

                List<Double> vcdM = batch.measurements().get("vcd");
                if (vcdM != null && i < vcdM.size() && vcdM.get(i) > 0) {
                    sumVCD += Math.abs((s.xv() - vcdM.get(i)) / vcdM.get(i));
                    countVCD++;
                }
                List<Double> tcdM = batch.measurements().get("tcd");
                if (tcdM != null && i < tcdM.size() && tcdM.get(i) > 0) {
                    sumTCD += Math.abs(((s.xv() + s.xd()) - tcdM.get(i)) / tcdM.get(i));
                    countTCD++;
                }
                List<Double> subM = batch.measurements().get("substrate");
                if (subM != null && i < subM.size() && subM.get(i) > 0) {
                    sumSub += Math.abs((s.substrate() - subM.get(i)) / subM.get(i));
                    countSub++;
                }
                List<Double> bypM = batch.measurements().get("byproduct");
                if (bypM != null && i < bypM.size() && bypM.get(i) > 0) {
                    sumByp += Math.abs((s.byproduct() - bypM.get(i)) / bypM.get(i));
                    countByp++;
                }
            }
        }

        return Map.of(
            "mape_vcd", sumVCD / Math.max(1, countVCD) * 100.0,
            "mape_tcd", sumTCD / Math.max(1, countTCD) * 100.0,
            "mape_substrate", sumSub / Math.max(1, countSub) * 100.0,
            "mape_byproduct", sumByp / Math.max(1, countByp) * 100.0
        );
    }
}

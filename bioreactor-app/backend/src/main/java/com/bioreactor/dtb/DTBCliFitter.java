package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import java.util.*;

/**
 * Quick CLI to evaluate DTB model on real data from PostgreSQL.
 */
public class DTBCliFitter {

    public static void main(String[] args) throws Exception {
        long start = System.currentTimeMillis();

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
        Map<String, Double> defaultParams = model.defaultParams();
        System.out.println("Default params: " + defaultParams);

        // Compute MAPE with RK4
        System.out.println("\n--- MAPE at default parameters (RK4, dt=0.01) ---");
        RK4Simulator rk4 = new RK4Simulator(model, 0.01);
        Map<String, Double> mape = computeMAPE(batches, defaultParams, rk4, model);
        System.out.println("MAPE: " + mape);

        long elapsed = System.currentTimeMillis() - start;
        System.out.println("\nElapsed: " + (elapsed / 1000.0) + " s");
        ds.close();
    }

    static Map<String, Double> computeMAPE(
        List<DTBDataReader.DTBBatch> batches, Map<String, Double> params,
        RK4Simulator rk4, DTBV2Model model
    ) {
        double sumVcd = 0, sumTcd = 0, sumSub = 0, sumByp = 0;
        int nVcd = 0, nTcd = 0, nSub = 0, nByp = 0;

        for (var b : batches) {
            var rr = rk4.simulate(b.initialState(), params, b.events(), b.measurementTimes().getLast());
            double[] ts = rr.times().stream().mapToDouble(Double::doubleValue).toArray();
            BioreactorState[] ss = rr.states().toArray(BioreactorState[]::new);

            for (int i = 0; i < b.measurementTimes().size(); i++) {
                double t = b.measurementTimes().get(i);
                var s = interpolate(ts, ss, t);
                if (s == null) continue;

                double oVcd = i < b.measurements().get("vcd").size() ? b.measurements().get("vcd").get(i) : Double.NaN;
                double oTcd = i < b.measurements().get("tcd").size() ? b.measurements().get("tcd").get(i) : Double.NaN;
                double oSub = i < b.measurements().get("substrate").size() ? b.measurements().get("substrate").get(i) : Double.NaN;
                double oByp = i < b.measurements().get("byproduct").size() ? b.measurements().get("byproduct").get(i) : Double.NaN;

                if (!Double.isNaN(oVcd) && oVcd > 1e-6) { sumVcd += relErr(s.xv(), oVcd); nVcd++; }
                if (!Double.isNaN(oTcd) && oTcd > 1e-6) { sumTcd += relErr(s.xv() + s.xd(), oTcd); nTcd++; }
                if (!Double.isNaN(oSub) && oSub > 1e-6) { sumSub += relErr(s.substrate(), oSub); nSub++; }
                if (!Double.isNaN(oByp) && oByp > 1e-6) { sumByp += relErr(s.byproduct(), oByp); nByp++; }
            }
        }

        return Map.of(
            "mape_vcd", pct(sumVcd, nVcd), "mape_tcd", pct(sumTcd, nTcd),
            "mape_substrate", pct(sumSub, nSub), "mape_byproduct", pct(sumByp, nByp)
        );
    }

    private static double relErr(double pred, double obs) { return Math.abs(pred - obs) / obs; }
    private static double pct(double sum, int n) { return n > 0 ? sum / n * 100 : Double.NaN; }

    private static BioreactorState interpolate(double[] ts, BioreactorState[] ss, double t) {
        if (t < ts[0] || t > ts[ts.length - 1]) return null;
        int lo = 0, hi = ts.length - 1;
        while (hi - lo > 1) { int m = (lo + hi) >>> 1; if (ts[m] <= t) lo = m; else hi = m; }
        if (Math.abs(ts[hi] - ts[lo]) < 1e-10) return ss[lo];
        double frac = (t - ts[lo]) / (ts[hi] - ts[lo]);
        BioreactorState s0 = ss[lo], s1 = ss[hi];
        return new BioreactorState(t,
            lerp(s0.xv(), s1.xv(), frac), lerp(s0.xd(), s1.xd(), frac),
            lerp(s0.substrate(), s1.substrate(), frac), lerp(s0.product(), s1.product(), frac),
            lerp(s0.byproduct(), s1.byproduct(), frac),
            s0.dissolvedO2(), s0.volume(), s0.temperature(), s0.pH());
    }
    private static double lerp(double a, double b, double f) { return a + f * (b - a); }
}

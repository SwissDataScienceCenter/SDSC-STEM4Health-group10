package com.bioreactor.dtb;

import com.bioreactor.engine.BioreactorState;
import com.bioreactor.engine.Simulator;

import javax.sql.DataSource;
import java.sql.*;
import java.util.*;

/**
 * Reads DTB experiment data from the database and converts it
 * to engine objects (BioreactorState, Simulator.Event).
 *
 * Works with the existing experiment_data table (wide format).
 */
public class DTBDataReader {

    private final DataSource dataSource;

    public DTBDataReader(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    /** A batch parsed from the database. */
    public record DTBBatch(
        String batchId,
        BioreactorState initialState,
        List<Simulator.Event> events,
        List<Double> measurementTimes,
        Map<String, List<Double>> measurements,  // parameter code -> values at measurementTimes
        boolean isTest
    ) {
        /** All measurement times including t=0. */
        public List<Double> allTimes() {
            List<Double> all = new ArrayList<>();
            all.add(0.0);
            all.addAll(measurementTimes);
            return all;
        }
    }

    /**
     * Load a single DTB batch. Time is in days.
     */
    public DTBBatch loadBatch(String batchId) throws SQLException {
        String sql = """
            SELECT ed.t, ed.vcd, ed.tcd, ed.substrate, ed.byproduct, ed.volume,
                   ed.feed_b, ed.feed_c, ed.feed_d, ed.to_predict
            FROM experiment_data ed
            JOIN experiments e ON ed.experiment_id = e.id
            WHERE e.project = 'DTB' AND e.batch_id = ?
            ORDER BY ed.t
            """;

        List<Double> tAll = new ArrayList<>();
        List<Double> vcdVals = new ArrayList<>();
        List<Double> tcdVals = new ArrayList<>();
        List<Double> subVals = new ArrayList<>();
        List<Double> bypVals = new ArrayList<>();
        List<Double> volVals = new ArrayList<>();
        List<Double> feedBTimes = new ArrayList<>();
        List<Double> feedBVals = new ArrayList<>();
        List<Double> feedCTimes = new ArrayList<>();
        List<Double> feedCVals = new ArrayList<>();
        List<Double> feedDTimes = new ArrayList<>();
        List<Double> feedDVals = new ArrayList<>();
        List<Double> measTimes = new ArrayList<>();
        boolean isTest = false;

        try (Connection c = dataSource.getConnection();
             PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, batchId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    double t = rs.getDouble("t");
                    double vcd = rs.getDouble("vcd");
                    boolean hasVcd = !rs.wasNull();
                    double tcd = rs.getDouble("tcd");
                    boolean hasTcd = !rs.wasNull();
                    double vol = rs.getDouble("volume");
                    double sub = rs.getDouble("substrate");
                    boolean hasSub = !rs.wasNull();
                    double byp = rs.getDouble("byproduct");
                    boolean hasByp = !rs.wasNull();

                    double feedB = rs.getDouble("feed_b");
                    boolean hasFeedB = !rs.wasNull();
                    double feedC = rs.getDouble("feed_c");
                    boolean hasFeedC = !rs.wasNull();
                    double feedD = rs.getDouble("feed_d");
                    boolean hasFeedD = !rs.wasNull();

                    isTest = rs.getBoolean("to_predict");

                    tAll.add(t);

                    if (hasVcd) {
                        measTimes.add(t);
                        vcdVals.add(vcd);
                        tcdVals.add(hasTcd ? tcd : Double.NaN);
                        subVals.add(hasSub ? sub : Double.NaN);
                        bypVals.add(hasByp ? byp : Double.NaN);
                        volVals.add(vol);
                    }

                    if (hasFeedB) { feedBTimes.add(t); feedBVals.add(feedB); }
                    if (hasFeedC) { feedCTimes.add(t); feedCVals.add(feedC); }
                    if (hasFeedD) { feedDTimes.add(t); feedDVals.add(feedD); }
                }
            }
        }

        // Build events from feed rows (feed volumes are already in L)
        // Filter out near-zero feed volumes (sensor noise)
        List<Simulator.Event> events = new ArrayList<>();
        for (int i = 0; i < feedBTimes.size(); i++) {
            double t = feedBTimes.get(i);
            double dV = feedBVals.get(i);
            if (dV > 0.01) events.add(Simulator.Event.feedBolus(t, dV, "feed_b"));
        }
        for (int i = 0; i < feedCTimes.size(); i++) {
            double t = feedCTimes.get(i);
            double dV = feedCVals.get(i);
            if (dV > 0.01) events.add(Simulator.Event.feedBolus(t, dV, "feed_c"));
        }
        for (int i = 0; i < feedDTimes.size(); i++) {
            double t = feedDTimes.get(i);
            double dV = feedDVals.get(i);
            if (dV > 0.01) events.add(Simulator.Event.feedBolus(t, dV, "feed_d"));
        }

        // Sampling events: set volume to measured volume (skip t=0, it's initial)
        for (int i = 0; i < measTimes.size(); i++) {
            double t = measTimes.get(i);
            if (t < 1e-10) continue;  // t=0 is initial state, not a volume reset
            double vol = volVals.get(i);
            events.add(Simulator.Event.sampleVolume(t, vol));
        }

        // Sort events by time
        events.sort(Comparator.comparingDouble(Simulator.Event::time));

        // Initial state from first measurement
        double vcd0 = vcdVals.isEmpty() ? 1.0 : vcdVals.get(0);
        double sub0 = subVals.isEmpty() ? 5.0 : subVals.get(0);
        double byp0 = bypVals.isEmpty() ? 0.0 : bypVals.get(0);
        double vol0 = volVals.isEmpty() ? 10.0 : volVals.get(0);

        BioreactorState ic = new BioreactorState(
            0, vcd0, 0, sub0, 0, byp0, 0.007, vol0, 37.0, 7.0
        );

        Map<String, List<Double>> measurements = Map.of(
            "vcd", vcdVals,
            "tcd", tcdVals,
            "substrate", subVals,
            "byproduct", bypVals
        );

        return new DTBBatch(batchId, ic, events, measTimes, measurements, isTest);
    }

    /** Load all training batches. */
    public List<DTBBatch> loadTrainingBatches() throws SQLException {
        List<DTBBatch> batches = new ArrayList<>();
        List<String> batchIds = getDTBBatchIds();
        for (String id : batchIds) {
            DTBBatch b = loadBatch(id);
            if (!b.isTest()) batches.add(b);
        }
        return batches;
    }

    /** Load all test batches. */
    public List<DTBBatch> loadTestBatches() throws SQLException {
        List<DTBBatch> batches = new ArrayList<>();
        List<String> batchIds = getDTBBatchIds();
        for (String id : batchIds) {
            DTBBatch b = loadBatch(id);
            if (b.isTest()) batches.add(b);
        }
        return batches;
    }

    private List<String> getDTBBatchIds() throws SQLException {
        List<String> ids = new ArrayList<>();
        try (Connection c = dataSource.getConnection();
             Statement s = c.createStatement();
             ResultSet rs = s.executeQuery(
                 "SELECT batch_id FROM experiments WHERE project = 'DTB' ORDER BY batch_id")) {
            while (rs.next()) ids.add(rs.getString("batch_id"));
        }
        return ids;
    }
}

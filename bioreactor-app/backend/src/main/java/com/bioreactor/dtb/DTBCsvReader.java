package com.bioreactor.dtb;

import com.bioreactor.engine.BioreactorState;
import com.bioreactor.engine.Simulator;

import java.io.*;
import java.util.*;

/**
 * Reads DTB experiment data from the CSV file directly (no PostgreSQL required).
 *
 * The CSV columns are: batch_id, elapsed_time, vcd, tcd, substrate, byproduct,
 * feed_substrate, feed_b, feed_c, feed_d, volume, to_predict
 *
 * This lets us fit on REAL data without needing the database.
 */
public class DTBCsvReader {

    private final String csvPath;

    public DTBCsvReader(String csvPath) {
        this.csvPath = csvPath;
    }

    /** A batch parsed from CSV. */
    public record DTBBatch(
        String batchId,
        BioreactorState initialState,
        List<Simulator.Event> events,
        List<Double> measurementTimes,
        Map<String, List<Double>> measurements,
        boolean isTest
    ) {}

    /** Load ALL batches from CSV. */
    public List<DTBBatch> loadAllBatches() throws IOException {
        Map<String, List<String[]>> byBatch = new LinkedHashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String header = br.readLine();
            String[] cols = header.split(",", -1);
            // Find column indices
            int idxId = find(cols, "batch_id");
            int idxT = find(cols, "elapsed_time");
            int idxVcd = find(cols, "vcd");
            int idxTcd = find(cols, "tcd");
            int idxSub = find(cols, "substrate");
            int idxByp = find(cols, "byproduct");
            int idxFeedB = find(cols, "feed_b");
            int idxFeedC = find(cols, "feed_c");
            int idxFeedD = find(cols, "feed_d");
            int idxFeedSub = find(cols, "feed_substrate");
            int idxVol = find(cols, "volume");
            int idxPred = find(cols, "to_predict");

            String line;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",", -1);
                String bid = row[idxId].trim();
                byBatch.computeIfAbsent(bid, k -> new ArrayList<>()).add(row);
            }
        }

        List<DTBBatch> batches = new ArrayList<>();
        for (var entry : byBatch.entrySet()) {
            String bid = entry.getKey();
            List<String[]> rows = entry.getValue();
            // Sort by time
            rows.sort(Comparator.comparingDouble(r -> parseDouble(r, 1)));

            // Find column indices from first row
            String[] cols = "batch_id,elapsed_time,vcd,tcd,substrate,byproduct,feed_substrate,feed_b,feed_c,feed_d,volume,to_predict".split(",", -1);
            int idxT = find(cols, "elapsed_time");
            int idxVcd = find(cols, "vcd");
            int idxTcd = find(cols, "tcd");
            int idxSub = find(cols, "substrate");
            int idxByp = find(cols, "byproduct");
            int idxFeedB = find(cols, "feed_b");
            int idxFeedC = find(cols, "feed_c");
            int idxFeedD = find(cols, "feed_d");
            int idxFeedSub = find(cols, "feed_substrate");
            int idxVol = find(cols, "volume");
            int idxPred = find(cols, "to_predict");

            List<Double> measTimes = new ArrayList<>();
            List<Double> vcdVals = new ArrayList<>();
            List<Double> tcdVals = new ArrayList<>();
            List<Double> subVals = new ArrayList<>();
            List<Double> bypVals = new ArrayList<>();
            List<Double> volVals = new ArrayList<>();
            boolean isTest = false;

            List<Simulator.Event> events = new ArrayList<>();

            for (String[] row : rows) {
                double t = parseDouble(row, idxT);
                double vcd = parseDouble(row, idxVcd);
                double tcd = parseDouble(row, idxTcd);
                double sub = parseDouble(row, idxSub);
                double byp = parseDouble(row, idxByp);
                double feedB = parseDouble(row, idxFeedB);
                double feedC = parseDouble(row, idxFeedC);
                double feedD = parseDouble(row, idxFeedD);
                double feedSub = parseDouble(row, idxFeedSub);
                double vol = parseDouble(row, idxVol);
                isTest = row[idxPred].trim().equals("True");

                // Record measurement if VCD present
                if (!Double.isNaN(vcd)) {
                    measTimes.add(t);
                    vcdVals.add(vcd);
                    tcdVals.add(tcd);
                    subVals.add(sub);
                    bypVals.add(byp);
                    volVals.add(vol);
                }

                // Feed events: use feed name so the model resolves concentrations
                // including hidden nutrient N in Feed B
                if (!Double.isNaN(feedB) && feedB > 0.01) {
                    events.add(Simulator.Event.feedBolus(t, feedB, "feed_b"));
                }
                if (!Double.isNaN(feedC) && feedC > 0.01) {
                    events.add(Simulator.Event.feedBolus(t, feedC, "feed_c"));
                }
                if (!Double.isNaN(feedD) && feedD > 0.01) {
                    events.add(Simulator.Event.feedBolus(t, feedD, "feed_d"));
                }
                // Substrate feed events (rare but present)
                if (!Double.isNaN(feedSub) && feedSub > 0.01) {
                    events.add(Simulator.Event.feedBolus(t, feedSub, "substrate"));
                }

                // Volume sampling events (skip t=0 as that's initial state)
                if (t > 1e-10 && !Double.isNaN(vol)) {
                    events.add(Simulator.Event.sampleVolume(t, vol));
                }
            }

            events.sort(Comparator.comparingDouble(Simulator.Event::time));

            // Initial conditions from first measurement
            double vcd0 = vcdVals.isEmpty() ? 1.0 : vcdVals.get(0);
            double sub0 = subVals.isEmpty() ? 5.0 : subVals.get(0);
            double byp0 = bypVals.isEmpty() ? 0.0 : bypVals.get(0);
            double vol0 = volVals.isEmpty() ? 10.0 : volVals.get(0);

            // Initial hidden states: [N=1.0 (normalized nutrient), D=0 (no damage)]
            double[] hidden = new double[]{1.0, 0.0};
            BioreactorState ic = new BioreactorState(
                0, vcd0, 0, sub0, 0, byp0, 0.007, vol0, 37.0, 7.0, hidden
            );

            Map<String, List<Double>> measurements = new HashMap<>();
            measurements.put("vcd", vcdVals);
            measurements.put("tcd", tcdVals);
            measurements.put("substrate", subVals);
            measurements.put("byproduct", bypVals);

            batches.add(new DTBBatch(bid, ic, events, measTimes, measurements, isTest));
        }
        return batches;
    }

    /** Load training batches (to_predict=False). */
    public List<DTBBatch> loadTrainingBatches() throws IOException {
        return loadAllBatches().stream().filter(b -> !b.isTest()).toList();
    }

    /** Load test batches (to_predict=True). */
    public List<DTBBatch> loadTestBatches() throws IOException {
        return loadAllBatches().stream().filter(DTBBatch::isTest).toList();
    }

    private static int find(String[] arr, String key) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].trim().equals(key)) return i;
        }
        return -1;
    }

    private static double parseDouble(String[] row, int idx) {
        if (idx < 0 || idx >= row.length) return Double.NaN;
        String val = row[idx].trim();
        if (val.isEmpty()) return Double.NaN;
        try {
            return Double.parseDouble(val);
        } catch (NumberFormatException e) {
            return Double.NaN;
        }
    }
}

package com.bioreactor.service;

import com.bioreactor.dtb.DTBCsvReader;
import com.bioreactor.engine.Simulator;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

/**
 * Imports DTB challenge CSV data into the V3 long-format data store.
 *
 * Reads bioreaction_remaining.csv via DTBCsvReader and populates:
 *   - run (one row per batch)
 *   - reading (time-series measurements per batch)
 *   - run_event (feed bolus and sample volume events)
 *
 * The process_definition row for 'DTB-v1' is created by Flyway V3.
 */
@Service
public class DTBImportService {

    private final JdbcTemplate jdbc;

    public DTBImportService(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    /** Import all batches from a CSV file. Clears existing DTB run data first. */
    @Transactional
    public int importAll(String csvPath) throws java.io.IOException {
        // Clear existing DTB run data (cascading)
        jdbc.update("DELETE FROM run_event WHERE run_id IN (SELECT id FROM run WHERE process_id = (SELECT id FROM process_definition WHERE code = 'DTB-v1'))");
        jdbc.update("DELETE FROM reading WHERE run_id IN (SELECT id FROM run WHERE process_id = (SELECT id FROM process_definition WHERE code = 'DTB-v1'))");
        jdbc.update("DELETE FROM prediction_value");
        jdbc.update("DELETE FROM prediction_run");
        jdbc.update("DELETE FROM run WHERE process_id = (SELECT id FROM process_definition WHERE code = 'DTB-v1')");

        UUID processId = jdbc.queryForObject(
            "SELECT id FROM process_definition WHERE code = 'DTB-v1'", UUID.class);

        // Read CSV
        var csvReader = new DTBCsvReader(csvPath);
        List<DTBCsvReader.DTBBatch> batches = csvReader.loadAllBatches();

        int totalReadings = 0;
        int totalEvents = 0;

        for (var batch : batches) {
            // Insert run
            UUID runId = UUID.randomUUID();
            String role = batch.isTest() ? "TEST" : "TRAIN";
            jdbc.update(
                "INSERT INTO run (id, process_id, code, kind, role) VALUES (?::uuid, ?::uuid, ?, 'DEVELOPMENT', ?)",
                runId.toString(), processId.toString(), batch.batchId(), role);

            // Insert readings at each measurement time
            var vcd = batch.measurements().getOrDefault("vcd", List.of());
            var tcd = batch.measurements().getOrDefault("tcd", List.of());
            var sub = batch.measurements().getOrDefault("substrate", List.of());
            var byp = batch.measurements().getOrDefault("byproduct", List.of());
            var vol = batch.measurements().getOrDefault("volume", List.of());

            for (int i = 0; i < batch.measurementTimes().size(); i++) {
                double t = batch.measurementTimes().get(i);
                if (i < vcd.size() && !Double.isNaN(vcd.get(i)))
                    insertReading(runId, t, "vcd", vcd.get(i), "10^6 cells/mL");
                if (i < tcd.size() && !Double.isNaN(tcd.get(i)))
                    insertReading(runId, t, "tcd", tcd.get(i), "10^6 cells/mL");
                if (i < sub.size() && !Double.isNaN(sub.get(i)))
                    insertReading(runId, t, "substrate", sub.get(i), "g/L");
                if (i < byp.size() && !Double.isNaN(byp.get(i)))
                    insertReading(runId, t, "byproduct", byp.get(i), "g/L");
                if (i < vol.size() && !Double.isNaN(vol.get(i)))
                    insertReading(runId, t, "volume", vol.get(i), "L");
                totalReadings += 5;
            }

            // Insert events (feed bolus, sample volume)
            for (var ev : batch.events()) {
                String kind = switch (ev.type()) {
                    case FEED_BOLUS -> "FEED_BOLUS";
                    case SAMPLE_VOLUME -> "SAMPLE";
                };
                String materialCode = null;
                Double amount = null;
                if (ev.type() == Simulator.Event.EventType.FEED_BOLUS) {
                    amount = ev.dV();
                    materialCode = ev.feedName() != null ? ev.feedName() : "feed";
                } else if (ev.type() == Simulator.Event.EventType.SAMPLE_VOLUME) {
                    amount = ev.measuredVolume();
                }
                jdbc.update(
                    "INSERT INTO run_event (id, run_id, elapsed_time, kind, material_code, amount, unit) " +
                    "VALUES (?::uuid, ?::uuid, ?, ?, ?, ?, ?)",
                    UUID.randomUUID().toString(), runId.toString(), ev.time(),
                    kind, materialCode, amount, kind.equals("FEED_BOLUS") ? "L" : "L");
                totalEvents++;
            }
        }

        return batches.size();
    }

    private void insertReading(UUID runId, double t, String paramCode, double value, String unit) {
        jdbc.update(
            "INSERT INTO reading (id, run_id, parameter_code, elapsed_time, numeric_value, unit) " +
            "VALUES (?::uuid, ?::uuid, ?, ?, ?, ?)",
            UUID.randomUUID().toString(), runId.toString(), paramCode, t, value, unit);
    }

    /** Get DTB batch IDs from the database. */
    public List<String> getDTBBatchIds() {
        return jdbc.queryForList(
            "SELECT r.code FROM run r JOIN process_definition pd ON r.process_id = pd.id WHERE pd.code = 'DTB-v1' ORDER BY r.code",
            String.class);
    }

    /** Count rows. */
    public int getReadingCount() {
        Integer count = jdbc.queryForObject(
            "SELECT COUNT(*) FROM reading r JOIN run ru ON r.run_id = ru.id JOIN process_definition pd ON ru.process_id = pd.id WHERE pd.code = 'DTB-v1'",
            Integer.class);
        return count != null ? count : 0;
    }

    public int getRunCount() {
        Integer count = jdbc.queryForObject(
            "SELECT COUNT(*) FROM run WHERE process_id = (SELECT id FROM process_definition WHERE code = 'DTB-v1')",
            Integer.class);
        return count != null ? count : 0;
    }

    public int getEventCount() {
        Integer count = jdbc.queryForObject(
            "SELECT COUNT(*) FROM run_event re JOIN run r ON re.run_id = r.id WHERE r.process_id = (SELECT id FROM process_definition WHERE code = 'DTB-v1')",
            Integer.class);
        return count != null ? count : 0;
    }
}

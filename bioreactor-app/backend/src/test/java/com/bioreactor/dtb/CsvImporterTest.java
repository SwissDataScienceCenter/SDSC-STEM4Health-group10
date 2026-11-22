package com.bioreactor.dtb;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Importer round-trip: write a minimal CSV, read it back, verify column values.
 */
class CsvImporterTest {

    @TempDir
    Path tempDir;

    @Test
    void roundTripCsv() throws Exception {
        // Write a minimal CSV with known values
        String csv = """
            batch_id,elapsed_time,vcd,tcd,substrate,byproduct,feed_b,feed_c,feed_d,feed_substrate,volume,to_predict
            B1,0.0,0.5,0.5,10.0,0.0,,,,,10.0,False
            B1,24.0,1.5,1.6,8.0,0.5,0.1,,,0.3,9.9,False
            B1,48.0,3.0,3.1,5.0,1.0,,,,,9.8,False
            B2,0.0,0.6,0.6,12.0,0.0,,,,,10.0,True
            B2,24.0,2.0,2.1,7.0,0.8,0.15,,,0.25,9.9,True
            B2,48.0,4.0,4.1,3.0,1.5,,,,,9.8,True
            """;

        Path csvFile = tempDir.resolve("test.csv");
        Files.writeString(csvFile, csv);

        var reader = new DTBCsvReader(csvFile.toString());

        // Load all batches
        var allBatches = reader.loadAllBatches();
        assertEquals(2, allBatches.size(), "Should load 2 batches");

        // ── Batch B1 ──
        var b1 = allBatches.get(0);
        assertEquals("B1", b1.batchId());
        assertFalse(b1.isTest(), "B1 should be training (to_predict=False)");

        // Initial state
        assertEquals(0.5, b1.initialState().xv(), 1e-9);
        assertEquals(10.0, b1.initialState().substrate(), 1e-9);
        assertEquals(0.0, b1.initialState().byproduct(), 1e-9);
        assertEquals(10.0, b1.initialState().volume(), 1e-9);

        // Measurement times and values
        assertEquals(List.of(0.0, 24.0, 48.0), b1.measurementTimes());
        assertEquals(List.of(0.5, 1.5, 3.0), b1.measurements().get("vcd"));
        assertEquals(List.of(0.5, 1.6, 3.1), b1.measurements().get("tcd"));
        assertEquals(List.of(10.0, 8.0, 5.0), b1.measurements().get("substrate"));
        assertEquals(List.of(0.0, 0.5, 1.0), b1.measurements().get("byproduct"));

        // Events: feed bolus at 24h (Feed B + substrate feed), plus volume sampling
        assertEquals(4, b1.events().size(), "B1 should have 4 events (2 feed boluses + 2 volume samples)");

        // ── Batch B2 ──
        var b2 = allBatches.get(1);
        assertTrue(b2.isTest(), "B2 should be test (to_predict=True)");
        assertFalse(b2.measurementTimes().isEmpty());

        // ── Filtered loads ──
        var train = reader.loadTrainingBatches();
        assertEquals(1, train.size());
        assertEquals("B1", train.get(0).batchId());

        var test = reader.loadTestBatches();
        assertEquals(1, test.size());
        assertEquals("B2", test.get(0).batchId());
    }

    @Test
    void emptyCsvIsGraceful() throws Exception {
        String csv = """
            batch_id,elapsed_time,vcd,tcd,substrate,byproduct,feed_b,feed_c,feed_d,feed_substrate,volume,to_predict
            """;

        Path csvFile = tempDir.resolve("empty.csv");
        Files.writeString(csvFile, csv);

        var reader = new DTBCsvReader(csvFile.toString());
        var batches = reader.loadAllBatches();
        assertTrue(batches.isEmpty(), "Empty CSV should produce 0 batches");
    }

    @Test
    void missingColumnsHandled() throws Exception {
        // CSV missing some optional columns — reader should handle gracefully via find() returning -1
        String csv = """
            batch_id,elapsed_time,vcd,tcd,substrate,byproduct,feed_b,feed_c,feed_d,feed_substrate,volume,to_predict
            B1,0.0,0.5,0.5,10.0,0.0,,,,,10.0,False
            """;

        Path csvFile = tempDir.resolve("missing_cols.csv");
        Files.writeString(csvFile, csv);

        var reader = new DTBCsvReader(csvFile.toString());
        var batches = reader.loadAllBatches();
        assertEquals(1, batches.size());
        assertNotNull(batches.get(0).initialState());
    }
}

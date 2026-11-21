package com.bioreactor.dtb;

import com.bioreactor.engine.*;
import org.springframework.stereotype.Service;

import javax.sql.DataSource;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.sql.SQLException;
import java.util.*;

/**
 * Service for fitting DTB parameters and generating predictions.
 */
@Service
public class DTBPredictionService {

    private final DTBDataReader reader;
    private final Fitter fitter;
    private final ProcessModel model;

    public DTBPredictionService(DataSource dataSource) {
        this.reader = new DTBDataReader(dataSource);
        this.model = new DTBModel();
        this.fitter = new Fitter(model);
    }

    /**
     * Fit model on all training batches.
     */
    public Fitter.FitResult fitOnTrainingData() throws SQLException {
        List<DTBDataReader.DTBBatch> trainBatches = reader.loadTrainingBatches();
        List<Fitter.TrainingBatch> fitBatches = toFitterBatches(trainBatches);
        return fitter.fit(fitBatches, 200);
    }

    /**
     * Generate predictions for all test batches.
     */
    public List<TrialOutput> generatePredictions(Map<String, Double> params) throws SQLException {
        List<DTBDataReader.DTBBatch> testBatches = reader.loadTestBatches();
        List<TrialOutput> outputs = new ArrayList<>();

        for (var batch : testBatches) {
            var pred = fitter.predict(toFitterBatch(batch), params);

            var output = new TrialOutput(batch.batchId(), pred, batch.measurementTimes());
            outputs.add(output);
        }
        return outputs;
    }

    /**
     * Write predictions to a CSV file.
     */
    public File writePredictionCSV(List<TrialOutput> outputs) throws IOException {
        File f = File.createTempFile("dtb_predictions_", ".csv");

        try (PrintWriter w = new PrintWriter(f, StandardCharsets.UTF_8)) {
            w.println("batch_id,t,vcd,tcd,substrate,byproduct");
            for (var out : outputs) {
                for (var row : out.rows()) {
                    w.printf("%s,%.6f,%.6f,%.6f,%.6f,%.6f%n",
                        out.batchId(), row.time(), row.vcd(), row.tcd(),
                        row.substrate(), row.byproduct());
                }
            }
        }
        return f;
    }

    private List<Fitter.TrainingBatch> toFitterBatches(List<DTBDataReader.DTBBatch> batches) {
        return batches.stream().map(this::toFitterBatch).toList();
    }

    private Fitter.TrainingBatch toFitterBatch(DTBDataReader.DTBBatch b) {
        return new Fitter.TrainingBatch(
            b.batchId(), b.initialState(), b.events(),
            b.measurementTimes(),
            b.measurements().getOrDefault("vcd", List.of()),
            b.measurements().getOrDefault("substrate", List.of()),
            b.measurements().getOrDefault("byproduct", List.of())
        );
    }

    /** A single row of output. */
    public record TrialOutput(
        String batchId,
        Simulator.SimulationResult result,
        List<Double> targetTimes
    ) {
        /** Interpolate simulation to the target (measurement) times. */
        public List<PredictionRow> rows() {
            List<PredictionRow> r = new ArrayList<>();
            for (double t : targetTimes) {
                if (t == 0) {
                    var s = result.states().getFirst();
                    r.add(new PredictionRow(0, s.xv(), s.xv() + s.xd(),
                        s.substrate(), s.byproduct()));
                } else {
                    var s = interpolate(t);
                    r.add(new PredictionRow(t, s.xv(), s.xv() + s.xd(),
                        Math.max(0, s.substrate()), Math.max(0, s.byproduct())));
                }
            }
            return r;
        }

        private BioreactorState interpolate(double t) {
            List<Double> ts = result.times();
            List<BioreactorState> ss = result.states();
            int idx = 0;
            for (int i = 0; i < ts.size(); i++) {
                if (ts.get(i) >= t) { idx = i; break; }
                idx = i;
            }
            // Linear interpolation between idx and idx+1
            if (idx >= ts.size() - 1) return ss.getLast();
            double t0 = ts.get(idx), t1 = ts.get(idx + 1);
            if (t1 - t0 < 1e-10) return ss.get(idx);
            double frac = (t - t0) / (t1 - t0);
            BioreactorState s0 = ss.get(idx), s1 = ss.get(idx + 1);
            double xv  = s0.xv()  + frac * (s1.xv()  - s0.xv());
            double xd  = s0.xd()  + frac * (s1.xd()  - s0.xd());
            double sub = s0.substrate() + frac * (s1.substrate() - s0.substrate());
            double byp = s0.byproduct() + frac * (s1.byproduct() - s0.byproduct());
            double vol = s0.volume() + frac * (s1.volume() - s0.volume());
            return new BioreactorState(t, xv, xd, sub, 0, byp, 0.007, vol,
                s0.temperature(), s0.pH());
        }
    }

    public record PredictionRow(double time, double vcd, double tcd,
                                double substrate, double byproduct) {}
}

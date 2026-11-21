package com.bioreactor.service;

import com.bioreactor.engine.*;
import com.bioreactor.engine.ConfigurableModel.OrganismConfig;
import com.bioreactor.model.*;
import com.bioreactor.monitoring.MonitoringEngine;
import com.bioreactor.monitoring.MonitoringEngine.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;

/**
 * Central orchestrator for manufacturing process monitoring.
 *
 * Given a manufacturing run (from the long-format data store), this service:
 *   1. Loads the batch's initial conditions, feed events, and offline measurements
 *   2. Simulates the expected trajectory using the organism's ConfigurableModel
 *   3. Compares measured vs predicted values (model mismatch detection)
 *   4. Checks against reference bands from training data (MonitoringEngine)
 *   5. Computes process trends (TrendingEngine) across batches
 *   6. Generates forward predictions (PredictionEngine) from current state
 *
 * This is the core of the production bioreactor manufacturing application —
 * fed-batch process analysis and deviation detection for any organism.
 */
@Service
public class ProcessRunner {

    private final JdbcTemplate jdbc;
    private final OrganismConfigLoader configLoader;

    public ProcessRunner(JdbcTemplate jdbc, OrganismConfigLoader configLoader) {
        this.jdbc = jdbc;
        this.configLoader = configLoader;
    }

    /**
     * Load readings for a given run, sorted by elapsed time.
     * Returns one map per time point with parameter_code → numeric_value.
     */
    public List<TimePoint> loadReadings(String runId) {
        return jdbc.query(
            "SELECT elapsed_time, parameter_code, numeric_value FROM reading WHERE run_id = ?::uuid ORDER BY elapsed_time, parameter_code",
            rs -> {
                List<TimePoint> points = new ArrayList<>();
                double currentTime = -1;
                TimePoint current = null;
                while (rs.next()) {
                    double t = rs.getDouble("elapsed_time");
                    if (t != currentTime || current == null) {
                        if (current != null) points.add(current);
                        current = new TimePoint(t, new HashMap<>());
                        currentTime = t;
                    }
                    current.values.put(rs.getString("parameter_code"), rs.getDouble("numeric_value"));
                }
                if (current != null) points.add(current);
                return points;
            }, runId);
    }

    /**
     * Build reference bands from all training runs for a given process.
     * Uses the monitoring engine to compute p10/p50/p90 on a common time grid.
     */
    public Map<String, ReferenceBand> buildReferenceBands(String processId) {
        // Get all training run IDs for this process
        List<String> runIds = jdbc.queryForList(
            "SELECT id FROM run WHERE process_id = ?::uuid AND role = 'TRAIN'",
            String.class, processId);

        if (runIds.size() < 3) return Map.of(); // need at least 3 runs for meaningful bands

        // Collect time series for each parameter
        String[] params = {"vcd", "tcd", "substrate", "byproduct"};
        Map<String, List<double[]>> timeMap = new HashMap<>();
        Map<String, List<double[]>> valueMap = new HashMap<>();
        for (String p : params) { timeMap.put(p, new ArrayList<>()); valueMap.put(p, new ArrayList<>()); }

        // Build common time grid (min start, max end across runs)
        double globalStart = Double.MAX_VALUE, globalEnd = 0;

        for (String runId : runIds) {
            List<TimePoint> readings = loadReadings(runId);
            if (readings.isEmpty()) continue;
            globalStart = Math.min(globalStart, readings.get(0).time);
            globalEnd = Math.max(globalEnd, readings.get(readings.size() - 1).time);

            for (String p : params) {
                double[] t = new double[readings.size()];
                double[] v = new double[readings.size()];
                for (int i = 0; i < readings.size(); i++) {
                    t[i] = readings.get(i).time;
                    v[i] = readings.get(i).get(p, Double.NaN);
                }
                timeMap.get(p).add(t);
                valueMap.get(p).add(v);
            }
        }

        // Common grid: 50 points evenly spaced
        int gridPoints = 50;
        double[] commonGrid = new double[gridPoints];
        for (int i = 0; i < gridPoints; i++) {
            commonGrid[i] = globalStart + i * (globalEnd - globalStart) / (gridPoints - 1);
        }

        Map<String, ReferenceBand> bands = new HashMap<>();
        for (String p : params) {
            if (timeMap.get(p).size() < 3) continue;
            bands.put(p, MonitoringEngine.buildBand(p, timeMap.get(p), valueMap.get(p), commonGrid));
        }
        return bands;
    }

    /**
     * Load feed events for a given run.
     */
    public List<FeedEvent> loadFeedEvents(String runId) {
        return jdbc.query(
            "SELECT elapsed_time, material_code, amount, unit FROM run_event WHERE run_id = ?::uuid AND kind = 'FEED_BOLUS' ORDER BY elapsed_time",
            (rs, i) -> new FeedEvent(
                rs.getDouble("elapsed_time"),
                rs.getString("material_code"),
                rs.getDouble("amount"),
                rs.getString("unit")),
            runId);
    }

    /**
     * List all manufacturing runs with metadata.
     */
    public List<Map<String, Object>> listRuns() {
        return jdbc.queryForList(
            "SELECT r.id, r.code, r.kind, r.role, r.created_at, pd.name as process_name " +
            "FROM run r JOIN process_definition pd ON r.process_id = pd.id " +
            "ORDER BY r.created_at DESC LIMIT 50");
    }

    /**
     * Load run metadata (process_id, kind, etc.).
     */
    public Map<String, Object> loadRunMeta(String runId) {
        return jdbc.queryForMap(
            "SELECT r.id, r.code, r.kind, r.role, r.process_id, p.name as process_name, p.time_unit " +
            "FROM run r JOIN process_definition p ON r.process_id = p.id WHERE r.id = ?::uuid", runId);
    }

    /**
     * Run a full monitoring analysis for one manufacturing batch.
     *
     * Steps:
     *  1. Load the organism config and build the model
     *  2. Load initial state from the first reading
     *  3. Build feed events from run_event table
     *  4. Simulate expected trajectory
     *  5. Compare measurements to simulation → model mismatch findings
     *  6. Check against reference bands → outside-band findings
     *  7. Report summary metrics
     */
    @Transactional(readOnly = true)
    public ProcessReport analyzeRun(String runId, String organismId) {
        var start = System.currentTimeMillis();

        // 1. Load organism
        OrganismConfig config = configLoader.load(organismId);
        ConfigurableModel model = new ConfigurableModel(config);
        Map<String, Double> params = new HashMap<>();
        for (ModelParameter mp : model.parameters()) params.put(mp.name(), mp.startValue());

        // 2. Load run data
        List<TimePoint> readings = loadReadings(runId);
        List<FeedEvent> feedEvents = loadFeedEvents(runId);

        if (readings.isEmpty()) {
            return new ProcessReport(runId, organismId, "NO_DATA", "No readings found for run", 0,
                List.of(), List.of(), List.of(), List.of(), System.currentTimeMillis() - start);
        }

        // 3. Initial state from first reading
        TimePoint first = readings.get(0);
        double xv0 = first.get("vcd", 0.5);
        double s0  = first.get("substrate", 10.0);
        double v0  = first.get("volume", 1.0);

        BioreactorState initialState = new BioreactorState(
            0.0, xv0, 0.0, s0, 0.0, 0.0,
            0.007, v0, config.tOpt(), config.pHOpt());

        // 4. Build Simulator events from feed events
        List<Simulator.Event> simEvents = new ArrayList<>();
        for (FeedEvent fe : feedEvents) {
            if (fe.amount > 0.01) {
                simEvents.add(Simulator.Event.feedBolus(fe.time, fe.amount, fe.materialCode));
            }
        }

        // 5. Run simulation
        double tEnd = readings.get(readings.size() - 1).time;
        RK4Simulator sim = new RK4Simulator(model, 0.01);
        RK4Simulator.SimResult simResult = sim.simulate(initialState, params, simEvents, tEnd);

        // 6. Build reference bands from training runs (if available)
        var runMeta = loadRunMeta(runId);
        String processId = runMeta != null ? runMeta.get("process_id").toString() : null;
        Map<String, ReferenceBand> bands = processId != null ? buildReferenceBands(processId) : Map.of();

        // 7. Compare: at each measurement time, check model mismatch AND reference band
        List<Deviation> deviations = new ArrayList<>();
        List<FindingInfo> findings = new ArrayList<>();

        for (TimePoint tp : readings) {
            BioreactorState simState = interpolate(simResult, tp.time);
            if (simState == null) continue;

            for (var entry : tp.values.entrySet()) {
                String code = entry.getKey();
                double measured = entry.getValue();
                double predicted = switch (code) {
                    case "vcd" -> simState.xv();
                    case "tcd" -> simState.xv() + simState.xd();
                    case "substrate" -> simState.substrate();
                    case "byproduct" -> simState.byproduct();
                    case "volume" -> simState.volume();
                    default -> Double.NaN;
                };
                if (Double.isNaN(predicted)) continue;

                double absError = measured - predicted;
                double relError = predicted > 0 ? absError / predicted * 100 : 0;

                // Model mismatch check
                if (Math.abs(relError) > 15.0) {
                    deviations.add(new Deviation(tp.time, code, measured, predicted,
                        absError, relError,
                        Math.abs(relError) > 30.0 ? "HIGH" : "MODERATE"));
                    findings.add(new FindingInfo(tp.time, code,
                        Math.abs(relError) > 30.0 ? "ALARM" : "WARNING",
                        "MODEL_MISMATCH",
                        String.format("%s=%.3f vs predicted %.3f (%.1f%% error)",
                            code, measured, predicted, relError)));
                }

                // Reference band check
                ReferenceBand band = bands.get(code);
                if (band != null) {
                    String bandMsg = band.check(tp.time, measured);
                    if (bandMsg != null) {
                        findings.add(new FindingInfo(tp.time, code, "WARNING",
                            "OUTSIDE_REFERENCE_BAND", bandMsg));
                    }
                }
            }
        }

        // 8. Summarize key metrics
        List<Metric> metrics = summarizeMetrics(readings, simResult, tEnd);

        long elapsed = System.currentTimeMillis() - start;
        return new ProcessReport(runId, organismId, "OK",
            "Analysis complete in " + elapsed + "ms", tEnd,
            readings, deviations, findings, metrics, elapsed);
    }

    /** Interpolate simulation state at a given time. */
    private BioreactorState interpolate(RK4Simulator.SimResult result, double time) {
        List<Double> times = result.times();
        List<BioreactorState> states = result.states();
        for (int i = 0; i < times.size() - 1; i++) {
            if (time >= times.get(i) && time <= times.get(i + 1)) {
                double t1 = times.get(i), t2 = times.get(i + 1);
                BioreactorState s1 = states.get(i), s2 = states.get(i + 1);
                double frac = (time - t1) / (t2 - t1);

                return new BioreactorState(time,
                    s1.xv() + frac * (s2.xv() - s1.xv()),
                    s1.xd() + frac * (s2.xd() - s1.xd()),
                    s1.substrate() + frac * (s2.substrate() - s1.substrate()),
                    s1.product() + frac * (s2.product() - s1.product()),
                    s1.byproduct() + frac * (s2.byproduct() - s1.byproduct()),
                    s1.dissolvedO2() + frac * (s2.dissolvedO2() - s1.dissolvedO2()),
                    s1.volume() + frac * (s2.volume() - s1.volume()),
                    s1.temperature(),
                    s1.pH());
            }
        }
        return null;
    }

    /** Extract summary metrics. */
    private List<Metric> summarizeMetrics(List<TimePoint> readings, RK4Simulator.SimResult sim, double tEnd) {
        List<Metric> metrics = new ArrayList<>();

        // Peak VCD from readings
        double peakXv = readings.stream()
            .mapToDouble(tp -> tp.get("vcd", 0))
            .max().orElse(0);
        metrics.add(new Metric("peak_vcd", peakXv, "10^6 cells/mL",
            "Max viable cell density across the batch"));

        // Final values
        TimePoint last = readings.get(readings.size() - 1);
        metrics.add(new Metric("final_vcd", last.get("vcd", 0), "10^6 cells/mL", "Final VCD"));
        metrics.add(new Metric("final_substrate", last.get("substrate", 0), "g/L", "Final substrate"));
        metrics.add(new Metric("duration", tEnd, "days", "Batch duration"));

        return metrics;
    }

    // ── Trending: control charts across batches ──

    /**
     * Compute trending analysis (I-MR control charts) for a set of batches.
     * Returns control chart data and violation descriptions.
     */
    public TrendReport trendAnalysis(String processId, String parameterCode) {
        List<String> runIds = jdbc.queryForList(
            "SELECT id FROM run WHERE process_id = ?::uuid AND role = 'TRAIN'",
            String.class, processId);

        if (runIds.isEmpty()) {
            return new TrendReport(parameterCode, 0.0, 0.0, 0.0, 0.0, 0.0, List.of(), List.of());
        }

        // Extract metric values (e.g. peak VCD) per batch
        double[] values = new double[runIds.size()];
        String[] batchIds = new String[runIds.size()];
        for (int i = 0; i < runIds.size(); i++) {
            batchIds[i] = runIds.get(i);
            var readings = loadReadings(runIds.get(i));
            values[i] = switch (parameterCode) {
                case "peak_vcd" -> readings.stream().mapToDouble(tp -> tp.get("vcd", 0)).max().orElse(0);
                case "final_vcd" -> {
                    var last = readings.get(readings.size() - 1);
                    yield last.get("vcd", 0);
                }
                case "harvest_titer" -> {
                    var last = readings.get(readings.size() - 1);
                    yield last.get("substrate", 0);
                }
                default -> readings.stream().mapToDouble(tp -> tp.get(parameterCode, 0)).average().orElse(0);
            };
        }

        // Build I-MR chart
        var chart = com.bioreactor.trending.TrendingEngine.buildIMRChart(
            parameterCode, batchIds, values);

        // Compute Cpk if possible
        double mean = chart.centreLine();
        double sigma = mean > 0 ? (chart.ucl() - mean) / 3.0 : 0;
        double cpk = sigma > 0 ? Math.min(mean / (3 * sigma), mean / (3 * sigma)) : 0;

        double[] vals = chart.values();
        List<Double> boxed = new ArrayList<>(vals.length);
        for (double v : vals) boxed.add(v);
        return new TrendReport(
            parameterCode, mean, sigma, chart.ucl(), chart.lcl(), cpk,
            boxed, chart.violations());
    }

    // ── Data types ──

    public record TimePoint(double time, Map<String, Double> values) {
        public double get(String code, double def) { return values.getOrDefault(code, def); }
    }

    public record FeedEvent(double time, String materialCode, double amount, String unit) {}

    public record Deviation(double time, String parameter, double measured, double predicted,
                            double absError, double relErrorPct, String severity) {}

    public record FindingInfo(double elapsedTime, String parameterCode, String severity,
                              String kind, String message) {}

    public record Metric(String code, double value, String unit, String description) {}

    public record ProcessReport(String runId, String organismId, String status, String message,
                                double tEnd, List<TimePoint> readings, List<Deviation> deviations,
                                List<FindingInfo> findings, List<Metric> metrics, long elapsedMs) {}

    public record TrendReport(String parameterCode, double mean, double sigma,
                              double ucl, double lcl, double cpk,
                              List<Double> values, List<String> violations) {}
}

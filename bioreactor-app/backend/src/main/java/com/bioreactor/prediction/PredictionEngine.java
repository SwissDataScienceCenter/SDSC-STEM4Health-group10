package com.bioreactor.prediction;

import com.bioreactor.engine.*;

import java.util.*;

/**
 * Prediction engine: takes a fitted model and generates predictions
 * for new batches from initial conditions + feed profile only.
 *
 * This is the core "predict" operation defined in the spec:
 * "Predict takes a model version, initial values, events, volume profile
 *  and times. Returns parameter_code, time, value, lower, upper and warnings."
 */
public class PredictionEngine {

    private final Simulator simulator;

    public PredictionEngine(ProcessModel model) {
        this.simulator = new Simulator(model);
    }

    public record PredictionInput(
        String batchId,
        BioreactorState initialState,
        List<Simulator.Event> events,
        List<Double> targetTimes,   // times to output predictions for
        double tEnd
    ) {}

    public record PredictionRow(
        String parameterCode,
        double time,
        double value,
        double lower,
        double upper
    ) {}

    public record Prediction(
        String batchId,
        List<PredictionRow> rows,
        List<String> warnings
    ) {}

    /**
     * Single-batch prediction.
     */
    public Prediction predict(
        PredictionInput input,
        Map<String, Double> params,
        Map<String, PredictionBand> bands
    ) {
        double tEnd = Math.max(input.tEnd(), input.targetTimes().isEmpty() ? 10 :
            Collections.max(input.targetTimes()) * 1.05);

        Simulator.SimulationResult sim = simulator.simulate(
            input.initialState(), params, input.events(), tEnd, 0.01);

        List<PredictionRow> rows = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        for (double t : input.targetTimes()) {
            BioreactorState s = interpolate(sim, t);
            double tcd = s.xv() + s.xd();
            double sub = Math.max(0, s.substrate());
            double byp = Math.max(0, s.byproduct());

            for (var entry : Map.of(
                "vcd", s.xv(),
                "tcd", tcd,
                "substrate", sub,
                "byproduct", byp
            ).entrySet()) {
                String code = entry.getKey();
                double val = entry.getValue();
                double lower = val;
                double upper = val;

                PredictionBand band = bands != null ? bands.get(code) : null;
                if (band != null) {
                    lower = val - band.absError(t);
                    upper = val + band.absError(t);
                }

                rows.add(new PredictionRow(code, t, val,
                    Math.max(0, lower), upper));
            }
        }

        if (bands == null || bands.isEmpty()) {
            warnings.add("No prediction band available; lower/upper = point estimate.");
        }

        return new Prediction(input.batchId(), rows, warnings);
    }

    /**
     * Error band from out-of-fold validation errors, per variable.
     */
    public record PredictionBand(
        String parameterCode,
        double[] times,       // reference times
        double[] absErrors    // absolute error at each time
    ) {
        public double absError(double t) {
            // nearest reference time
            if (t <= times[0]) return absErrors[0];
            if (t >= times[times.length - 1]) return absErrors[times.length - 1];
            for (int i = 0; i < times.length - 1; i++) {
                if (t >= times[i] && t < times[i + 1]) {
                    double frac = (t - times[i]) / (times[i + 1] - times[i]);
                    return absErrors[i] + frac * (absErrors[i + 1] - absErrors[i]);
                }
            }
            return absErrors[absErrors.length - 1];
        }

        /**
         * Build bands from out-of-fold prediction errors.
         * times: grid of elapsed times. errors: list of (pred - observed) per time.
         */
        public static PredictionBand fromErrors(
            String code, double[] times, List<Double> errors,
            double coverageFactor // e.g. 2.0 for ~95% CI
        ) {
            // Bin errors by nearest time
            double[] absErr = new double[times.length];
            int[] counts = new int[times.length];
            for (int i = 0; i < errors.size(); i++) {
                double t = times[i];
                int bin = 0;
                double minDist = Double.POSITIVE_INFINITY;
                for (int j = 0; j < times.length; j++) {
                    double d = Math.abs(t - times[j]);
                    if (d < minDist) { minDist = d; bin = j; }
                }
                absErr[bin] += Math.abs(errors.get(i));
                counts[bin]++;
            }
            for (int i = 0; i < times.length; i++) {
                absErr[i] = counts[i] > 0 ? absErr[i] / counts[i] * coverageFactor : 0;
            }
            // Fill gaps
            for (int i = 1; i < times.length; i++) {
                if (counts[i] == 0) absErr[i] = absErr[i - 1];
            }
            return new PredictionBand(code, times, absErr);
        }
    }

    private BioreactorState interpolate(Simulator.SimulationResult r, double t) {
        List<Double> ts = r.times();
        List<BioreactorState> ss = r.states();
        if (t <= ts.get(0)) return ss.get(0);
        if (t >= ts.get(ts.size() - 1)) return ss.get(ts.size() - 1);

        int idx = 0;
        for (int i = 0; i < ts.size() - 1; i++) {
            if (t >= ts.get(i) && t < ts.get(i + 1)) { idx = i; break; }
            idx = i;
        }
        if (idx >= ts.size() - 1) return ss.get(ts.size() - 1);

        double t0 = ts.get(idx), t1 = ts.get(idx + 1);
        double frac = (t - t0) / (t1 - t0);
        BioreactorState s0 = ss.get(idx), s1 = ss.get(idx + 1);

        return new BioreactorState(t,
            s0.xv() + frac * (s1.xv() - s0.xv()),
            s0.xd() + frac * (s1.xd() - s0.xd()),
            s0.substrate() + frac * (s1.substrate() - s0.substrate()),
            s0.product() + frac * (s1.product() - s0.product()),
            s0.byproduct() + frac * (s1.byproduct() - s0.byproduct()),
            s0.dissolvedO2(), s0.volume(), s0.temperature(), s0.pH());
    }
}

package com.bioreactor.engine;

import java.util.*;

/**
 * Parameter fitting using Levenberg-Marquardt or bounded optimization.
 * Fits a ProcessModel against experimental data across all training batches.
 */
public class Fitter {

    private final ProcessModel model;
    private final Simulator simulator;

    public Fitter(ProcessModel model) {
        this.model = model;
        this.simulator = new Simulator(model);
    }

    /**
     * A single batch of training data.
     */
    public record TrainingBatch(
        String id,
        BioreactorState initialState,
        List<Simulator.Event> events,
        List<Double> times,          // observation times
        List<Double> observedVCD,    // observed values at those times
        List<Double> observedSubstrate,
        List<Double> observedByproduct
    ) {}

    /**
     * Fit result.
     */
    public record FitResult(
        Map<String, Double> fittedParams,
        Map<String, Double> startParams,
        boolean converged,
        double finalCost,
        Map<String, Double> rSquared,
        int iterations
    ) {}

    /**
     * Fit model to all training batches simultaneously (shared parameters).
     * Uses a simple grid search + local refinement.
     */
    public FitResult fit(List<TrainingBatch> batches, int maxIterations) {
        List<ModelParameter> params = model.parameters();
        Map<String, Double> best = new HashMap<>();
        double bestCost = Double.POSITIVE_INFINITY;

        // Use start values as initial guess
        for (ModelParameter p : params) {
            best.put(p.name(), p.startValue());
        }

        // Simple Nelder-Mead-like search: perturb one parameter at a time
        double stepSize = 0.3;
        for (int iter = 0; iter < maxIterations; iter++) {
            double currentCost = computeCost(batches, best);

            Map<String, Double> candidate = new HashMap<>(best);
            boolean improved = false;

            for (ModelParameter p : params) {
                // Try +step
                double orig = candidate.get(p.name());
                candidate.put(p.name(), orig * (1 + stepSize));
                double costUp = computeCost(batches, candidate);

                if (costUp < currentCost) {
                    best.put(p.name(), candidate.get(p.name()));
                    currentCost = costUp;
                    improved = true;
                } else {
                    // Try -step
                    candidate.put(p.name(), orig * (1 - stepSize));
                    double costDown = computeCost(batches, candidate);
                    if (costDown < currentCost) {
                        best.put(p.name(), candidate.get(p.name()));
                        currentCost = costDown;
                        improved = true;
                    } else {
                        candidate.put(p.name(), orig);
                    }
                }
            }

            if (!improved) {
                stepSize *= 0.5;  // reduce step
            }
            if (stepSize < 0.001) break;
        }

        double finalCost = computeCost(batches, best);
        return new FitResult(best, null, true, finalCost, computeRSquared(batches, best), maxIterations);
    }

    private double computeCost(List<TrainingBatch> batches, Map<String, Double> params) {
        double total = 0;
        int n = 0;
        for (TrainingBatch b : batches) {
            double tEnd = b.times().isEmpty() ? 10 : b.times().getLast() * 1.05;
            var result = simulator.simulate(b.initialState(), params, b.events(), tEnd, 0.05);

            for (int i = 0; i < b.times().size() && i < b.observedVCD().size(); i++) {
                double t = b.times().get(i);
                if (t <= 0) continue;
                double observed = b.observedVCD().get(i);
                if (Double.isNaN(observed)) continue;

                // Find nearest simulated point
                double simulated = nearest(result.times(), result.states(), t);
                double relErr = (simulated - observed) / Math.max(0.5, Math.abs(observed));
                total += relErr * relErr;
                n++;
            }
        }
        return n > 0 ? Math.sqrt(total / n) : Double.POSITIVE_INFINITY;
    }

    private double nearest(List<Double> times, List<BioreactorState> states, double t) {
        int idx = 0;
        for (int i = 0; i < times.size(); i++) {
            if (times.get(i) >= t) { idx = i; break; }
            idx = i;
        }
        return states.get(idx).xv();
    }

    private Map<String, Double> computeRSquared(List<TrainingBatch> batches, Map<String, Double> params) {
        double ssRes = 0, ssTot = 0;
        int n = 0;
        double sumObs = 0;

        for (TrainingBatch b : batches) {
            double tEnd = b.times().isEmpty() ? 10 : b.times().getLast() * 1.05;
            var result = simulator.simulate(b.initialState(), params, b.events(), tEnd, 0.05);

            for (int i = 1; i < b.times().size() && i < b.observedVCD().size(); i++) {
                double obs = b.observedVCD().get(i);
                if (Double.isNaN(obs)) continue;
                double sim = nearest(result.times(), result.states(), b.times().get(i));
                double e = obs - sim;
                ssRes += e * e;
                sumObs += obs;
                n++;
            }
        }
        if (n == 0) return Map.of("R2_vcd", 0.0);

        double mean = sumObs / n;
        for (TrainingBatch b : batches) {
            for (int i = 1; i < b.times().size() && i < b.observedVCD().size(); i++) {
                double obs = b.observedVCD().get(i);
                if (Double.isNaN(obs)) continue;
                double d = obs - mean;
                ssTot += d * d;
            }
        }
        double r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
        return Map.of("R2_vcd", r2);
    }

    /** Predict a single batch from its initial state + events only. */
    public Simulator.SimulationResult predict(TrainingBatch batch, Map<String, Double> params) {
        double tEnd = batch.times().isEmpty() ? 10 : batch.times().getLast() * 1.05;
        return simulator.simulate(batch.initialState(), params, batch.events(), tEnd, 0.05);
    }

    /**
     * Compute MAPE between predicted and observed values.
     */
    public static Map<String, Double> computeMAPE(
        Simulator.SimulationResult predicted,
        List<Double> times,
        List<Double> observedVCD,
        List<Double> observedSubstrate,
        List<Double> observedByproduct
    ) {
        double sumVcd = 0, sumSub = 0, sumByp = 0;
        int nVcd = 0, nSub = 0, nByp = 0;

        for (int i = 1; i < times.size(); i++) {
            double t = times.get(i);
            double obsVcd = i < observedVCD.size() ? observedVCD.get(i) : Double.NaN;
            double obsSub = i < observedSubstrate.size() ? observedSubstrate.get(i) : Double.NaN;
            double obsByp = i < observedByproduct.size() ? observedByproduct.get(i) : Double.NaN;

            // Find nearest predicted states
            BioreactorState sim = nearestState(predicted, t);

            if (!Double.isNaN(obsVcd) && obsVcd > 0) {
                sumVcd += Math.abs((sim.xv() - obsVcd) / obsVcd);
                nVcd++;
            }
            if (!Double.isNaN(obsSub) && obsSub > 0) {
                sumSub += Math.abs((sim.substrate() - obsSub) / obsSub);
                nSub++;
            }
            if (!Double.isNaN(obsByp) && obsByp > 0) {
                sumByp += Math.abs((sim.byproduct() - obsByp) / obsByp);
                nByp++;
            }
        }

        return Map.of(
            "mape_vcd", nVcd > 0 ? sumVcd / nVcd * 100 : Double.NaN,
            "mape_substrate", nSub > 0 ? sumSub / nSub * 100 : Double.NaN,
            "mape_byproduct", nByp > 0 ? sumByp / nByp * 100 : Double.NaN
        );
    }

    private static BioreactorState nearestState(Simulator.SimulationResult r, double t) {
        int idx = 0;
        for (int i = 0; i < r.times().size(); i++) {
            if (r.times().get(i) >= t) { idx = i; break; }
            idx = i;
        }
        return r.states().get(idx);
    }
}

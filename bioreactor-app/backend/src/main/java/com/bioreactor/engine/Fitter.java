package com.bioreactor.engine;

import org.hipparchus.linear.*;
import org.hipparchus.optim.nonlinear.vector.leastsquares.LevenbergMarquardtOptimizer;
import org.hipparchus.optim.nonlinear.vector.leastsquares.LeastSquaresBuilder;
import org.hipparchus.optim.nonlinear.vector.leastsquares.LeastSquaresOptimizer;
import org.hipparchus.optim.nonlinear.vector.leastsquares.LeastSquaresProblem;
import org.hipparchus.optim.nonlinear.vector.leastsquares.MultivariateJacobianFunction;
import org.hipparchus.optim.nonlinear.vector.leastsquares.ParameterValidator;
import org.hipparchus.optim.nonlinear.vector.leastsquares.LeastSquaresOptimizer.Optimum;
import org.hipparchus.util.Pair;

import java.util.*;

/**
 * Parameter fitting using bounded Levenberg-Marquardt (Hipparchus).
 * Fits a ProcessModel on all training batches with shared parameters.
 *
 * Ported from Python: src/mechanistic.py fit() with f_scale / loss='linear'.
 * Loss follows the leaderboard MAPE metric (relative error on 4 variables).
 */
public class Fitter {

    private final ProcessModel model;
    private final Simulator simulator;

    public Fitter(ProcessModel model) {
        this.model = model;
        this.simulator = new Simulator(model);
    }

    /** A single batch of training data. */
    public record TrainingBatch(
        String id,
        BioreactorState initialState,
        List<Simulator.Event> events,
        List<Double> times,             // observation times (excluding t=0)
        List<Double> observedVCD,       // observed values at those times
        List<Double> observedTCD,       // total cell density
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
        Map<String, Double> mapePerVariable,
        int iterations
    ) {}

    /** Map between parameter names and array indices. */
    private static record ParamIndex(String name, int idx) {}

    /**
     * Fit model to all training batches simultaneously.
     * Uses bounded Levenberg-Marquardt starting from the model's default parameter values.
     */
    public FitResult fit(List<TrainingBatch> batches, int maxIterations) {
        List<ModelParameter> spec = model.parameters();
        List<ParamIndex> indices = new ArrayList<>();
        for (int i = 0; i < spec.size(); i++) {
            indices.add(new ParamIndex(spec.get(i).name(), i));
        }

        double[] start = new double[spec.size()];
        double[] lower = new double[spec.size()];
        double[] upper = new double[spec.size()];
        for (int i = 0; i < spec.size(); i++) {
            ModelParameter p = spec.get(i);
            start[i] = p.toUnconstrained(p.startValue());
            lower[i] = p.toUnconstrained(p.lowerBound());
            upper[i] = p.toUnconstrained(p.upperBound());
        }

        // Collect all (predicted, observed) pairs for residuals
        List<ResidualPoint> points = collectPoints(batches);

        MultivariateJacobianFunction modelFunc = point -> {
            double[] p = new double[spec.size()];
            for (int i = 0; i < spec.size(); i++) {
                p[i] = spec.get(i).fromUnconstrained(point.getEntry(i));
            }

            Map<String, Double> params = arrayToMap(spec, p);
            double[] residuals = computeResiduals(batches, params, points);
            double[][] jacobian = computeJacobian(spec, params, batches, points);

            return new Pair<>(new ArrayRealVector(residuals),
                              new Array2DRowRealMatrix(jacobian));
        };

        // Build the least-squares problem
        LeastSquaresBuilder builder = new LeastSquaresBuilder();
        builder.model(modelFunc);
        builder.target(new double[points.size()]);
        builder.start(start);
        builder.maxIterations(maxIterations);
        builder.maxEvaluations(Math.max(maxIterations * spec.size(), 500));

        // Use Levenberg-Marquardt with parameter bounds
        ParameterValidator validator = new ParameterValidator() {
            @Override public RealVector validate(RealVector params) {
                double[] bounded = new double[spec.size()];
                for (int i = 0; i < spec.size(); i++) {
                    bounded[i] = Math.min(upper[i], Math.max(lower[i], params.getEntry(i)));
                }
                return new ArrayRealVector(bounded);
            }
        };
        builder.parameterValidator(validator);

        LeastSquaresOptimizer optimizer = new LevenbergMarquardtOptimizer();
        LeastSquaresProblem problem = builder.build();
        Optimum optimum;
        try {
            optimum = optimizer.optimize(problem);
        } catch (org.hipparchus.exception.MathIllegalStateException e) {
            // Max iterations/evaluations exceeded — use the last point
            optimum = optimizer.optimize(problem);
            // If we can't even get an optimum, evaluate at start point
            double[] startFitted = new double[spec.size()];
            for (int i = 0; i < spec.size(); i++)
                startFitted[i] = spec.get(i).fromUnconstrained(start[i]);
            Map<String, Double> startMap = arrayToMap(spec, startFitted);
            return new FitResult(startMap, startMap, false,
                Double.POSITIVE_INFINITY, computeMAPE(batches, startMap), 0);
        }

        // Map fitted params back to natural scale
        double[] fitted = new double[spec.size()];
        for (int i = 0; i < spec.size(); i++) {
            fitted[i] = spec.get(i).fromUnconstrained(optimum.getPoint().getEntry(i));
        }

        Map<String, Double> fittedMap = arrayToMap(spec, fitted);
        Map<String, Double> mape = computeMAPE(batches, fittedMap);

        return new FitResult(
            fittedMap,
            arrayToMap(spec, start),
            optimum.getIterations() < maxIterations,
            optimum.getRMS(),
            mape,
            optimum.getIterations()
        );
    }

    /**
     * Multi-start fit from several random initial points, taking the best result.
     */
    public FitResult fitMultiStart(List<TrainingBatch> batches, int restarts, int maxIterPerStart) {
        FitResult best = null;
        double bestCost = Double.POSITIVE_INFINITY;

        // Always try the nominal start
        var nom = fit(batches, maxIterPerStart);
        if (nom.converged() && nom.finalCost() < bestCost) {
            best = nom;
            bestCost = nom.finalCost();
        }

        // Random restarts
        List<ModelParameter> spec = model.parameters();
        Random rng = new Random(42);
        for (int r = 0; r < restarts; r++) {
            // Perturb start values within bounds
            double[] perturbed = new double[spec.size()];
            for (int i = 0; i < spec.size(); i++) {
                ModelParameter p = spec.get(i);
                double range = p.upperBound() - p.lowerBound();
                double val = p.startValue() + (rng.nextDouble() - 0.5) * range * 0.5;
                val = Math.min(p.upperBound(), Math.max(p.lowerBound(), val));
                perturbed[i] = p.toUnconstrained(val);
            }

            var result = fitFromStart(batches, maxIterPerStart, spec, perturbed);
            if (result.converged() && result.finalCost() < bestCost) {
                best = result;
                bestCost = result.finalCost();
            }
        }
        return best;
    }

    private FitResult fitFromStart(List<TrainingBatch> batches, int maxIterations,
                                    List<ModelParameter> spec, double[] start) {
        double[] lower = new double[spec.size()];
        double[] upper = new double[spec.size()];
        for (int i = 0; i < spec.size(); i++) {
            lower[i] = spec.get(i).toUnconstrained(spec.get(i).lowerBound());
            upper[i] = spec.get(i).toUnconstrained(spec.get(i).upperBound());
        }

        List<ResidualPoint> points = collectPoints(batches);

        MultivariateJacobianFunction modelFunc = point -> {
            double[] p = new double[spec.size()];
            for (int i = 0; i < spec.size(); i++) {
                p[i] = spec.get(i).fromUnconstrained(point.getEntry(i));
            }
            Map<String, Double> params = arrayToMap(spec, p);
            double[] residuals = computeResiduals(batches, params, points);
            double[][] jacobian = computeJacobian(spec, params, batches, points);
            return new Pair<>(new ArrayRealVector(residuals),
                              new Array2DRowRealMatrix(jacobian));
        };

        LeastSquaresBuilder builder = new LeastSquaresBuilder();
        builder.model(modelFunc);
        builder.target(new double[points.size()]);
        builder.start(start);
        builder.maxIterations(maxIterations);
        builder.maxEvaluations(Math.max(maxIterations * spec.size(), 500));
        builder.parameterValidator(params -> {
            double[] bounded = new double[spec.size()];
            for (int i = 0; i < spec.size(); i++)
                bounded[i] = Math.min(upper[i], Math.max(lower[i], params.getEntry(i)));
            return new ArrayRealVector(bounded);
        });

        Optimum optimum = new LevenbergMarquardtOptimizer()
            .optimize(builder.build());

        double[] fitted = new double[spec.size()];
        for (int i = 0; i < spec.size(); i++)
            fitted[i] = spec.get(i).fromUnconstrained(optimum.getPoint().getEntry(i));

        Map<String, Double> fittedMap = arrayToMap(spec, fitted);
        return new FitResult(fittedMap, arrayToMap(spec, start),
            optimum.getIterations() < maxIterations, optimum.getRMS(),
            computeMAPE(batches, fittedMap), optimum.getIterations());
    }

    /**
     * Leave-one-batch-out cross-validation.
     */
    public Map<String, Double> crossValidate(List<TrainingBatch> batches, int maxIterPerFold) {
        double totalVcd = 0, totalTcd = 0, totalSub = 0, totalByp = 0;
        int nVcd = 0, nTcd = 0, nSub = 0, nByp = 0;

        for (int i = 0; i < batches.size(); i++) {
            TrainingBatch heldOut = batches.get(i);
            List<TrainingBatch> train = new ArrayList<>(batches);
            train.remove(i);

            FitResult fit = fit(train, maxIterPerFold);
            Simulator.SimulationResult pred = predict(heldOut, fit.fittedParams());

            for (int j = 1; j < heldOut.times().size(); j++) {
                double t = heldOut.times().get(j);
                double obsVcd = j < heldOut.observedVCD().size() ? heldOut.observedVCD().get(j) : Double.NaN;
                double obsTcd = j < heldOut.observedTCD().size() ? heldOut.observedTCD().get(j) : Double.NaN;
                double obsSub = j < heldOut.observedSubstrate().size() ? heldOut.observedSubstrate().get(j) : Double.NaN;
                double obsByp = j < heldOut.observedByproduct().size() ? heldOut.observedByproduct().get(j) : Double.NaN;

                BioreactorState sim = nearestState(pred, t);

                if (!Double.isNaN(obsVcd) && obsVcd > 1e-6) {
                    totalVcd += Math.abs((sim.xv() - obsVcd) / obsVcd);
                    nVcd++;
                }
                if (!Double.isNaN(obsTcd) && obsTcd > 1e-6) {
                    double tcdPred = sim.xv() + sim.xd();
                    totalTcd += Math.abs((tcdPred - obsTcd) / obsTcd);
                    nTcd++;
                }
                if (!Double.isNaN(obsSub) && obsSub > 1e-6) {
                    totalSub += Math.abs((sim.substrate() - obsSub) / obsSub);
                    nSub++;
                }
                if (!Double.isNaN(obsByp) && obsByp > 1e-6) {
                    totalByp += Math.abs((sim.byproduct() - obsByp) / obsByp);
                    nByp++;
                }
            }
        }
        return Map.of(
            "mape_vcd", nVcd > 0 ? totalVcd / nVcd * 100 : Double.NaN,
            "mape_tcd", nTcd > 0 ? totalTcd / nTcd * 100 : Double.NaN,
            "mape_substrate", nSub > 0 ? totalSub / nSub * 100 : Double.NaN,
            "mape_byproduct", nByp > 0 ? totalByp / nByp * 100 : Double.NaN
        );
    }

    /** Predict a single batch from its initial state + events only. */
    public Simulator.SimulationResult predict(TrainingBatch batch, Map<String, Double> params) {
        double tEnd = batch.times().isEmpty() ? 10 : batch.times().getLast() * 1.05;
        return simulator.simulate(batch.initialState(), params, batch.events(), tEnd, 0.05);
    }

    // --- private helpers ---

    private record ResidualPoint(
        String batchId, double t,
        double observedVcd, double observedTcd,
        double observedSub, double observedByp
    ) {}

    private List<ResidualPoint> collectPoints(List<TrainingBatch> batches) {
        List<ResidualPoint> points = new ArrayList<>();
        double scaleVcd = 10.0, scaleTcd = 10.0, scaleSub = 2.0, scaleByp = 1.5;

        for (var b : batches) {
            for (int i = 1; i < b.times().size(); i++) {
                double obsVcd = i < b.observedVCD().size() ? b.observedVCD().get(i) : Double.NaN;
                double obsTcd = i < b.observedTCD().size() ? b.observedTCD().get(i) : Double.NaN;
                double obsSub = i < b.observedSubstrate().size() ? b.observedSubstrate().get(i) : Double.NaN;
                double obsByp = i < b.observedByproduct().size() ? b.observedByproduct().get(i) : Double.NaN;

                if (!Double.isNaN(obsVcd) && obsVcd > 1e-6)
                    points.add(new ResidualPoint(b.id(), b.times().get(i),
                        obsVcd, Double.NaN, Double.NaN, Double.NaN));
                if (!Double.isNaN(obsTcd) && obsTcd > 1e-6)
                    points.add(new ResidualPoint(b.id(), b.times().get(i),
                        Double.NaN, obsTcd, Double.NaN, Double.NaN));
                if (!Double.isNaN(obsSub) && obsSub > 1e-6)
                    points.add(new ResidualPoint(b.id(), b.times().get(i),
                        Double.NaN, Double.NaN, obsSub, Double.NaN));
                if (!Double.isNaN(obsByp) && obsByp > 1e-6)
                    points.add(new ResidualPoint(b.id(), b.times().get(i),
                        Double.NaN, Double.NaN, Double.NaN, obsByp));
            }
        }
        return points;
    }

    /** Weighted relative error: (pred - obs)/max(|obs|, epsilon) */
    private double[] computeResiduals(List<TrainingBatch> batches,
                                       Map<String, Double> params,
                                       List<ResidualPoint> points) {
        // Simulate all batches once
        Map<String, Simulator.SimulationResult> simResults = new HashMap<>();
        for (var b : batches)
            simResults.put(b.id(), predict(b, params));

        double[] residuals = new double[points.size()];
        double eps = 0.5; // floor for relative error denominator

        for (int i = 0; i < points.size(); i++) {
            ResidualPoint pt = points.get(i);
            Simulator.SimulationResult sim = simResults.get(pt.batchId());
            if (sim == null) { residuals[i] = 0; continue; }

            BioreactorState s = nearestState(sim, pt.t());

            if (!Double.isNaN(pt.observedVcd())) {
                double denom = Math.max(Math.abs(pt.observedVcd()), eps);
                residuals[i] = (s.xv() - pt.observedVcd()) / denom;
            } else if (!Double.isNaN(pt.observedTcd())) {
                double denom = Math.max(Math.abs(pt.observedTcd()), eps);
                double tcdPred = s.xv() + s.xd();
                residuals[i] = (tcdPred - pt.observedTcd()) / denom;
            } else if (!Double.isNaN(pt.observedSub())) {
                double denom = Math.max(Math.abs(pt.observedSub()), eps);
                residuals[i] = (s.substrate() - pt.observedSub()) / denom;
            } else if (!Double.isNaN(pt.observedByp())) {
                double denom = Math.max(Math.abs(pt.observedByp()), eps);
                residuals[i] = (s.byproduct() - pt.observedByp()) / denom;
            }
        }
        return residuals;
    }

    /** Finite-difference Jacobian for the residuals. */
    private double[][] computeJacobian(List<ModelParameter> spec,
                                        Map<String, Double> params,
                                        List<TrainingBatch> batches,
                                        List<ResidualPoint> points) {
        int m = points.size();
        int n = spec.size();
        double[][] J = new double[m][n];
        double[] baseResiduals = computeResiduals(batches, params, points);
        double h = 1e-6;

        for (int j = 0; j < n; j++) {
            ModelParameter p = spec.get(j);
            double orig = params.get(p.name());
            double perturb = orig * (1 + h);
            if (perturb == orig) perturb = orig + h;
            perturb = Math.min(p.upperBound(), Math.max(p.lowerBound(), perturb));

            Map<String, Double> perturbedParams = new HashMap<>(params);
            perturbedParams.put(p.name(), perturb);
            double[] perturbedResiduals = computeResiduals(batches, perturbedParams, points);

            for (int i = 0; i < m; i++) {
                J[i][j] = (perturbedResiduals[i] - baseResiduals[i]) / (perturb - orig);
            }
        }
        return J;
    }

    private Map<String, Double> computeMAPE(List<TrainingBatch> batches,
                                            Map<String, Double> params) {
        double sumVcd = 0, sumTcd = 0, sumSub = 0, sumByp = 0;
        int nVcd = 0, nTcd = 0, nSub = 0, nByp = 0;

        for (var b : batches) {
            var pred = predict(b, params);
            for (int i = 1; i < b.times().size(); i++) {
                double t = b.times().get(i);
                BioreactorState sim = nearestState(pred, t);
                double obsVcd = i < b.observedVCD().size() ? b.observedVCD().get(i) : Double.NaN;
                double obsTcd = i < b.observedTCD().size() ? b.observedTCD().get(i) : Double.NaN;
                double obsSub = i < b.observedSubstrate().size() ? b.observedSubstrate().get(i) : Double.NaN;
                double obsByp = i < b.observedByproduct().size() ? b.observedByproduct().get(i) : Double.NaN;

                if (!Double.isNaN(obsVcd) && obsVcd > 1e-6) {
                    sumVcd += Math.abs((sim.xv() - obsVcd) / obsVcd);
                    nVcd++;
                }
                if (!Double.isNaN(obsTcd) && obsTcd > 1e-6) {
                    sumTcd += Math.abs(((sim.xv() + sim.xd()) - obsTcd) / obsTcd);
                    nTcd++;
                }
                if (!Double.isNaN(obsSub) && obsSub > 1e-6) {
                    sumSub += Math.abs((sim.substrate() - obsSub) / obsSub);
                    nSub++;
                }
                if (!Double.isNaN(obsByp) && obsByp > 1e-6) {
                    sumByp += Math.abs((sim.byproduct() - obsByp) / obsByp);
                    nByp++;
                }
            }
        }
        return Map.of(
            "mape_vcd", nVcd > 0 ? sumVcd / nVcd * 100 : Double.NaN,
            "mape_tcd", nTcd > 0 ? sumTcd / nTcd * 100 : Double.NaN,
            "mape_substrate", nSub > 0 ? sumSub / nSub * 100 : Double.NaN,
            "mape_byproduct", nByp > 0 ? sumByp / nByp * 100 : Double.NaN
        );
    }

    private Map<String, Double> arrayToMap(List<ModelParameter> spec, double[] values) {
        Map<String, Double> map = new HashMap<>();
        for (int i = 0; i < spec.size(); i++)
            map.put(spec.get(i).name(), values[i]);
        return map;
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

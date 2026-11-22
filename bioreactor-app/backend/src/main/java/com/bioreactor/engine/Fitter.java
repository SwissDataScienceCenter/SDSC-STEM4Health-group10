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
 * Ported from Python: src/mechanistic.py fit(), simulate(), with f_scale / loss='linear'.
 * Loss follows the leaderboard MAPE metric (relative error on 4 variables).
 *
 * Uses fixed-step RK4 (dt=0.01, like the Python team's simulator) for speed during fitting.
 * The adaptive DormandPrince853 Simulator is still available for high-accuracy single runs.
 */
public class Fitter {

    private final ProcessModel model;
    private final RK4Simulator rk4;

    public Fitter(ProcessModel model) {
        this.model = model;
        this.rk4 = new RK4Simulator(model, 0.01);
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
     *
     * @param maxFunctionEvals maximum number of cost+Jacobian evaluations (like Python max_nfev)
     *        Each LM iteration does 1 + N_params evaluations so 500 means ~20-25 iterations
     */
    public FitResult fit(List<TrainingBatch> batches, int maxFunctionEvals) {
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
        // Pre-compute per-variable scales (don't change between LM iterations)
        double[] scales = computeScales(batches);

        MultivariateJacobianFunction modelFunc = point -> {
            double[] p = new double[spec.size()];
            for (int i = 0; i < spec.size(); i++) {
                p[i] = spec.get(i).fromUnconstrained(point.getEntry(i));
            }

            Map<String, Double> params = arrayToMap(spec, p);
            double[] residuals = computeResiduals(batches, params, points, scales);
            double[][] jacobian = computeJacobian(spec, params, batches, points, scales);

            return new Pair<>(new ArrayRealVector(residuals),
                              new Array2DRowRealMatrix(jacobian));
        };

        // Hipparchus Levenberg-Marquardt:
        // maxIterations = max LM outer loop iterations (controls restart attempts)
        // maxEvaluations = max cost+Jacobian evaluations (the real limit)
        // We set both high and let the internal convergence criteria stop earlier.
        LeastSquaresBuilder builder = new LeastSquaresBuilder();
        builder.model(modelFunc);
        builder.target(new double[points.size()]);
        builder.start(start);
        builder.maxIterations(maxFunctionEvals);
        builder.maxEvaluations(maxFunctionEvals);

        // Use Levenberg-Marquardt with parameter bounds
        ParameterValidator validator = params -> {
            double[] bounded = new double[spec.size()];
            for (int i = 0; i < spec.size(); i++) {
                bounded[i] = Math.min(upper[i], Math.max(lower[i], params.getEntry(i)));
            }
            return new ArrayRealVector(bounded);
        };
        builder.parameterValidator(validator);

        LeastSquaresOptimizer optimizer = new LevenbergMarquardtOptimizer();
        LeastSquaresProblem problem = builder.build();
        Optimum optimum;
        try {
            optimum = optimizer.optimize(problem);
        } catch (org.hipparchus.exception.MathIllegalStateException e) {
            System.err.println("WARNING: Fitter hit evaluation limit (" + maxFunctionEvals + ")");
            // start is unconstrained; convert to natural scale for the fallback result
            double[] startNatural = new double[spec.size()];
            for (int i = 0; i < spec.size(); i++)
                startNatural[i] = spec.get(i).fromUnconstrained(start[i]);
            Map<String, Double> sm = arrayToMap(spec, startNatural);
            return new FitResult(sm, sm, false,
                Double.POSITIVE_INFINITY, computeMAPE(batches, sm), 0);
        }

        // Map fitted params back to natural scale
        double[] fitted = new double[spec.size()];
        for (int i = 0; i < spec.size(); i++) {
            fitted[i] = spec.get(i).fromUnconstrained(optimum.getPoint().getEntry(i));
        }

        Map<String, Double> fittedMap = arrayToMap(spec, fitted);
        Map<String, Double> mape = computeMAPE(batches, fittedMap);

        boolean converged = optimum.getIterations() < maxFunctionEvals
            && optimum.getEvaluations() < maxFunctionEvals;
        return new FitResult(
            fittedMap,
            arrayToMap(spec, start),
            converged,
            optimum.getRMS(),
            mape,
            optimum.getIterations()
        );
    }

    /**
     * Multi-start fit from several random initial points, taking the best result.
     */
    public FitResult fitMultiStart(List<TrainingBatch> batches, int restarts, int maxEvalsPerStart) {
        FitResult best = null;
        double bestCost = Double.POSITIVE_INFINITY;

        // Always try the nominal start
        var nom = fit(batches, maxEvalsPerStart);
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

            var result = fitFromStart(batches, maxEvalsPerStart, spec, perturbed);
            if (result.converged() && result.finalCost() < bestCost) {
                best = result;
                bestCost = result.finalCost();
            }
        }
        return best;
    }

    private FitResult fitFromStart(List<TrainingBatch> batches, int maxFunctionEvals,
                                    List<ModelParameter> spec, double[] start) {
        double[] lower = new double[spec.size()];
        double[] upper = new double[spec.size()];
        for (int i = 0; i < spec.size(); i++) {
            lower[i] = spec.get(i).toUnconstrained(spec.get(i).lowerBound());
            upper[i] = spec.get(i).toUnconstrained(spec.get(i).upperBound());
        }

        List<ResidualPoint> points = collectPoints(batches);
        double[] scales = computeScales(batches);

        MultivariateJacobianFunction modelFunc = point -> {
            double[] p = new double[spec.size()];
            for (int i = 0; i < spec.size(); i++) {
                p[i] = spec.get(i).fromUnconstrained(point.getEntry(i));
            }
            Map<String, Double> params = arrayToMap(spec, p);
            double[] residuals = computeResiduals(batches, params, points, scales);
            double[][] jacobian = computeJacobian(spec, params, batches, points, scales);
            return new Pair<>(new ArrayRealVector(residuals),
                              new Array2DRowRealMatrix(jacobian));
        };

        LeastSquaresBuilder builder = new LeastSquaresBuilder();
        builder.model(modelFunc);
        builder.target(new double[points.size()]);
        builder.start(start);
        builder.maxIterations(maxFunctionEvals);
        builder.maxEvaluations(maxFunctionEvals);
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
        boolean converged = optimum.getIterations() < maxFunctionEvals
            && optimum.getEvaluations() < maxFunctionEvals;
        return new FitResult(fittedMap, arrayToMap(spec, start),
            converged, optimum.getRMS(),
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
            RK4Simulator.SimResult pred = predict(heldOut, fit.fittedParams());

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

    /** Predict a single batch from its initial state + events only. Uses fixed-step RK4. */
    public RK4Simulator.SimResult predict(TrainingBatch batch, Map<String, Double> params) {
        // Use the later of: last measurement time or last event time
        double lastMeasured = batch.times().isEmpty() ? 0 : batch.times().getLast();
        double lastEvent = batch.events().stream()
            .mapToDouble(Simulator.Event::time).max().orElse(0);
        double tMax = Math.max(lastMeasured, lastEvent);
        double tEnd = tMax > 1e-6 ? tMax * 1.05 : 10;

        // Ensure initial state has hidden states matching the model
        BioreactorState ic = batch.initialState();
        if (ic.hiddenCount() != model.hiddenStateCount()) {
            ic = new BioreactorState(ic.time(), ic.xv(), ic.xd(),
                ic.substrate(), ic.product(), ic.byproduct(),
                ic.dissolvedO2(), ic.volume(), ic.temperature(), ic.pH(),
                model.defaultHiddenState());
        }
        return rk4.simulate(ic, params, batch.events(), tEnd);
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

    /**
     * Scaled linear residuals: (pred - obs) / scale for each variable.
     * Uses per-variable standard deviation as the scale (Python default residual="scaled").
     * This preserves the natural weighting of each variable — VCD (range 0-30)
     * contributes more absolute error than substrate (range 0-5).
     */
    private double[] computeResiduals(List<TrainingBatch> batches,
                                       Map<String, Double> params,
                                       List<ResidualPoint> points) {
        // Compute variable scales from the training observations
        double[] scales = computeScales(batches);
        return computeResiduals(batches, params, points, scales);
    }

    private double[] computeResiduals(List<TrainingBatch> batches,
                                       Map<String, Double> params,
                                       List<ResidualPoint> points,
                                       double[] scales) {
        // Simulate all batches once
        Map<String, RK4Simulator.SimResult> simResults = new HashMap<>();
        for (var b : batches)
            simResults.put(b.id(), predict(b, params));

        double[] residuals = new double[points.size()];
        double maxResidual = 100.0;  // cap extreme residuals to prevent instability

        for (int i = 0; i < points.size(); i++) {
            ResidualPoint pt = points.get(i);
            RK4Simulator.SimResult sim = simResults.get(pt.batchId());
            if (sim == null) { residuals[i] = 0; continue; }

            BioreactorState s = nearestState(sim, pt.t());
            if (s == null) { residuals[i] = maxResidual; continue; }
            double pred, obs, scale;

            if (!Double.isNaN(pt.observedVcd())) {
                pred = s.xv(); obs = pt.observedVcd(); scale = scales[0];
            } else if (!Double.isNaN(pt.observedTcd())) {
                pred = s.xv() + s.xd(); obs = pt.observedTcd(); scale = scales[1];
            } else if (!Double.isNaN(pt.observedSub())) {
                pred = s.substrate(); obs = pt.observedSub(); scale = scales[2];
            } else if (!Double.isNaN(pt.observedByp())) {
                pred = s.byproduct(); obs = pt.observedByp(); scale = scales[3];
            } else {
                residuals[i] = 0;
                continue;
            }
            if (scale < 1e-6) scale = 1e-6;
            residuals[i] = Math.max(-maxResidual, Math.min(maxResidual, (pred - obs) / scale));
        }
        return residuals;
    }

    /** Compute per-variable standard deviations from training observations. */
    private double[] computeScales(List<TrainingBatch> batches) {
        double sumVcd = 0, sumTcd = 0, sumSub = 0, sumByp = 0;
        double sum2Vcd = 0, sum2Tcd = 0, sum2Sub = 0, sum2Byp = 0;
        int nVcd = 0, nTcd = 0, nSub = 0, nByp = 0;
        for (var b : batches) {
            for (int i = 0; i < b.times().size(); i++) {
                double v = i < b.observedVCD().size() ? b.observedVCD().get(i) : Double.NaN;
                double t = i < b.observedTCD().size() ? b.observedTCD().get(i) : Double.NaN;
                double s = i < b.observedSubstrate().size() ? b.observedSubstrate().get(i) : Double.NaN;
                double p = i < b.observedByproduct().size() ? b.observedByproduct().get(i) : Double.NaN;
                if (!Double.isNaN(v)) { sumVcd += v; sum2Vcd += v * v; nVcd++; }
                if (!Double.isNaN(t)) { sumTcd += t; sum2Tcd += t * t; nTcd++; }
                if (!Double.isNaN(s)) { sumSub += s; sum2Sub += s * s; nSub++; }
                if (!Double.isNaN(p)) { sumByp += p; sum2Byp += p * p; nByp++; }
            }
        }
        return new double[] {
            nVcd > 1 ? Math.sqrt((sum2Vcd - sumVcd * sumVcd / nVcd) / (nVcd - 1)) : 1.0,
            nTcd > 1 ? Math.sqrt((sum2Tcd - sumTcd * sumTcd / nTcd) / (nTcd - 1)) : 1.0,
            nSub > 1 ? Math.sqrt((sum2Sub - sumSub * sumSub / nSub) / (nSub - 1)) : 1.0,
            nByp > 1 ? Math.sqrt((sum2Byp - sumByp * sumByp / nByp) / (nByp - 1)) : 1.0
        };
    }

    /** Finite-difference Jacobian for the residuals. */
    private double[][] computeJacobian(List<ModelParameter> spec,
                                        Map<String, Double> params,
                                        List<TrainingBatch> batches,
                                        List<ResidualPoint> points,
                                        double[] scales) {
        int m = points.size();
        int n = spec.size();
        double[][] J = new double[m][n];
        double[] baseResiduals = computeResiduals(batches, params, points, scales);

        for (int j = 0; j < n; j++) {
            ModelParameter p = spec.get(j);
            double orig = params.get(p.name());
            // Use a relative + absolute perturbation: 1% of parameter range, min 1e-3
            double h = Math.max(1e-3, (p.upperBound() - p.lowerBound()) * 0.01);
            double perturb = Math.min(p.upperBound(), orig + h);
            if (perturb <= orig) perturb = Math.max(p.lowerBound(), orig - h);

            Map<String, Double> perturbedParams = new HashMap<>(params);
            perturbedParams.put(p.name(), perturb);
            double[] perturbedResiduals = computeResiduals(batches, perturbedParams, points, scales);

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

    private static BioreactorState nearestState(RK4Simulator.SimResult r, double t) {
        int idx = 0;
        for (int i = 0; i < r.times().size(); i++) {
            if (r.times().get(i) >= t) { idx = i; break; }
            idx = i;
        }
        return r.states().get(idx);
    }
}

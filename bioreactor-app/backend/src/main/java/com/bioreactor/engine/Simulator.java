package com.bioreactor.engine;

import org.hipparchus.ode.*;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;

import java.util.*;

/**
 * Simulator that integrates a ProcessModel between events.
 *
 * Ported from Python: simulation_engine.run_simulation, with the feed event bug fixed.
 */
public class Simulator {

    private final ProcessModel model;

    public Simulator(ProcessModel model) {
        this.model = model;
    }

    public record Event(double time, EventType type, double dV,
                        Map<String, Double> feedConcentrations,
                        Double measuredVolume) {
        public enum EventType { FEED_BOLUS, SAMPLE_VOLUME }
        public static Event feedBolus(double t, double dV, Map<String, Double> c) {
            return new Event(t, EventType.FEED_BOLUS, dV, c, null);
        }
        public static Event sampleVolume(double t, double V) {
            return new Event(t, EventType.SAMPLE_VOLUME, 0, Map.of(), V);
        }
    }

    public record SimulationResult(List<Double> times, List<BioreactorState> states) {}

    /**
     * Simulate to tEnd, applying events and recording at outputStep intervals.
     */
    public SimulationResult simulate(
        BioreactorState initialState, Map<String, Double> params,
        List<Event> events, double tEnd, double outputStep
    ) {
        List<Event> sorted = new ArrayList<>(events);
        sorted.sort(Comparator.comparingDouble(Event::time));

        List<Double> times = new ArrayList<>();
        List<BioreactorState> states = new ArrayList<>();

        BioreactorState current = initialState;
        double t = 0.0;
        int eventIdx = 0;

        times.add(t);
        states.add(current);

        while (t < tEnd - 1e-10) {
            // Apply events at current time
            while (eventIdx < sorted.size() && sorted.get(eventIdx).time() <= t + 1e-10) {
                Event ev = sorted.get(eventIdx);
                switch (ev.type()) {
                    case FEED_BOLUS:
                        current = current.applyBolus(ev.dV(), ev.feedConcentrations());
                        break;
                    case SAMPLE_VOLUME:
                        if (ev.measuredVolume() != null)
                            current = current.withVolume(ev.measuredVolume());
                        break;
                }
                eventIdx++;
            }

            // Next target: next event or next output step or tEnd
            double nextEvent = (eventIdx < sorted.size())
                ? sorted.get(eventIdx).time() : tEnd;
            double nextOutput = t + outputStep;
            double segmentEnd = Math.min(Math.min(nextEvent, nextOutput), tEnd);

            if (segmentEnd <= t + 1e-10) {
                t = Math.nextUp(t);
                continue;
            }

            // Integrate one step
            current = integrate(current, params, t, segmentEnd);
            t = segmentEnd;

            // Record
            if (t >= nextOutput - 1e-10 || t >= tEnd - 1e-10) {
                times.add(t);
                states.add(current);
            }
        }

        return new SimulationResult(times, states);
    }

    /**
     * Integrate from t0 to tf using Dormand-Prince 8(5,3).
     */
    public static BioreactorState integrate(
        BioreactorState state, Map<String, Double> params, double t0, double tf
    ) {
        if (tf <= t0) return state;

        OrdinaryDifferentialEquation ode = new OrdinaryDifferentialEquation() {
            @Override public int getDimension() { return 9; }
            @Override public double[] computeDerivatives(double t, double[] y) {
                BioreactorState s = BioreactorState.fromStateVector(t, y);

                // Use the MonodModel rates directly here (simplified for now)
                double mu_max  = params.getOrDefault("mu_max", 0.70);
                double K_s     = params.getOrDefault("K_s", 0.02);
                double Y_XS    = params.getOrDefault("Y_XS", 0.50);
                double m_s     = params.getOrDefault("m_s", 0.025);
                double alpha   = params.getOrDefault("alpha", 0.0);
                double beta    = params.getOrDefault("beta", 0.02);
                double k_d_val = params.getOrDefault("k_d", 0.01);
                double k_lys   = params.getOrDefault("k_lysis", 0.005);

                double T_min = 8.0, T_opt = 37.0, T_max = 45.0;
                double pH_opt = 7.0, pH_sigma = 0.8;

                double f_T = EnvironmentFactors.tempFactorCardinal(s.temperature(), T_min, T_opt, T_max);
                double f_pH = EnvironmentFactors.phFactorGaussian(s.pH(), pH_opt, pH_sigma);

                double S = Math.max(0, s.substrate());
                double mu = mu_max * (S / (K_s + S)) * f_T * f_pH;
                double q_S = (Y_XS > 0 ? mu / Y_XS : 0) + m_s;
                double q_B = alpha * mu + beta;

                double k_d_act = k_d_val * Math.max(0, 1.0 - S / (S + K_s));
                double dXv = (mu - k_d_act) * s.xv();
                double dXd = k_d_act * s.xv() - k_lys * s.xd();
                double dS_v = -q_S * s.xv();
                double dP = 0.0;
                double dB = q_B * s.xv();
                double dV = 0.0;
                double dT = 0.0;

                return new double[]{dXv, dXd, dS_v, dP, dB, 0.0, dV, dT, 0.0};
            }
        };

        DormandPrince853Integrator integrator = new DormandPrince853Integrator(
            1e-10, 1.0, 1e-8, 1e-8);

        ODEState initState = new ODEState(t0, state.stateVector());
        ODEState finalState = integrator.integrate(ode, initState, tf);

        return BioreactorState.fromStateVector(tf, finalState.getPrimaryState());
    }
}

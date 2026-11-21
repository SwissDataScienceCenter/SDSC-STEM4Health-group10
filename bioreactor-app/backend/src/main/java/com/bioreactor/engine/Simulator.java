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
                        String feedName,
                        Map<String, Double> feedConcentrations,
                        Double measuredVolume) {
        public enum EventType { FEED_BOLUS, SAMPLE_VOLUME }
        /** Create a feed bolus event. feedName identifies the feed stream (e.g. "feed_b", "substrate"). */
        public static Event feedBolus(double t, double dV, String feedName) {
            return new Event(t, EventType.FEED_BOLUS, dV, feedName, null, null);
        }
        /** Backward-compatible: feed with explicit concentrations. */
        public static Event feedBolus(double t, double dV, Map<String, Double> c) {
            return new Event(t, EventType.FEED_BOLUS, dV, null, c, null);
        }
        public static Event sampleVolume(double t, double V) {
            return new Event(t, EventType.SAMPLE_VOLUME, 0, null, Map.of(), V);
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
                    case FEED_BOLUS: {
                        Map<String, Double> conc = ev.feedConcentrations();
                        if (conc == null && ev.feedName() != null) {
                            // Look up from model: the model knows what each feed carries
                            conc = model.feedStreamConcentrations(ev.feedName(), params);
                        }
                        if (conc == null) conc = Map.of();
                        current = current.applyBolus(ev.dV(), conc);
                        break;
                    }
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
     * Delegates to the ProcessModel for the right-hand side rates.
     */
    public BioreactorState integrate(
        BioreactorState state, Map<String, Double> params, double t0, double tf
    ) {
        if (tf <= t0) return state;
        int dim = state.stateSize();

        OrdinaryDifferentialEquation ode = new OrdinaryDifferentialEquation() {
            @Override public int getDimension() { return dim; }
            @Override public double[] computeDerivatives(double t, double[] y) {
                BioreactorState s = BioreactorState.fromStateVector(t, y, model.hiddenStateCount());
                return model.rates(s, params);
            }
        };

        DormandPrince853Integrator integrator = new DormandPrince853Integrator(
            1e-10, 1.0, 1e-8, 1e-8);

        ODEState initState = new ODEState(t0, state.stateVector());
        ODEState finalState = integrator.integrate(ode, initState, tf);

        return BioreactorState.fromStateVector(tf, finalState.getPrimaryState(),
                                                model.hiddenStateCount());
    }
}

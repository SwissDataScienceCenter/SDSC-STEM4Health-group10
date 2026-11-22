package com.bioreactor.engine;

import java.util.*;

/**
 * Fixed-step Runge-Kutta 4 simulator — much faster than adaptive Dormand-Prince
 * for fitting on many batches with many events.
 *
 * Uses dt=0.01 as the team's Python does. Events are applied at grid boundaries
 * (timing error ≤ dt). Between events, volume is constant.
 *
 * Ported from Python: src/mechanistic.py simulate()
 * Ordering: RK4 step → volume reset → feed boluses (matched to Python per-step logic)
 */
public class RK4Simulator {

    private final ProcessModel model;
    private final double dt;  // fixed step size

    public RK4Simulator(ProcessModel model, double dt) {
        this.model = model;
        this.dt = dt;
    }

    public RK4Simulator(ProcessModel model) {
        this(model, 0.01);
    }

    public record SimResult(List<Double> times, List<BioreactorState> states) {}

    /**
     * Simulate using fixed-step RK4, applying events at grid boundaries.
     * Matches Python simulate(): RK4 step → volume reset → feed bolus.
     */
    public SimResult simulate(
        BioreactorState initialState, Map<String, Double> params,
        List<Simulator.Event> events, double tEnd
    ) {
        int nSteps = (int) Math.ceil(tEnd / dt) + 2;
        List<Simulator.Event> sorted = new ArrayList<>(events);
        sorted.sort(Comparator.comparingDouble(Simulator.Event::time));

        // Build event index: for each step k, list of events with time in (k*dt, (k+1)*dt]
        // In Python: step_of(t) = max(ceil(t/dt)-1, 0)
        // Then events at step k are applied AFTER the RK4 step k→k+1
        @SuppressWarnings("unchecked")
        List<Simulator.Event>[] stepEvents = new List[nSteps];
        for (int i = 0; i < nSteps; i++) stepEvents[i] = new ArrayList<>();
        for (var ev : sorted) {
            int k = Math.max(0, (int) Math.ceil(ev.time() / dt) - 1);
            if (k < nSteps) stepEvents[k].add(ev);
        }

        double[] x = initialState.stateVector();
        double V = initialState.volume();
        int hiddenCount = model.hiddenStateCount();

        // Use undiluted indices from the model
        Set<Integer> undilutedSet = model.undilutedHiddenIndices();

        // Dense output at dt resolution
        List<Double> times = new ArrayList<>();
        List<BioreactorState> states = new ArrayList<>();
        times.add(0.0);
        states.add(BioreactorState.fromStateVector(0.0, x, hiddenCount).withVolume(V));

        for (int k = 0; k < nSteps; k++) {
            double t = k * dt;

            // 1. RK4 step: x at k → x at k+1 (before events)
            if (k < nSteps - 1) {
                double[] k1 = model.rates(BioreactorState.fromStateVector(t, x, hiddenCount).withVolume(V), params);
                double[] x2 = add(x, scale(k1, 0.5 * dt));
                double[] k2 = model.rates(BioreactorState.fromStateVector(t + 0.5 * dt, x2, hiddenCount).withVolume(V), params);
                double[] x3 = add(x, scale(k2, 0.5 * dt));
                double[] k3 = model.rates(BioreactorState.fromStateVector(t + 0.5 * dt, x3, hiddenCount).withVolume(V), params);
                double[] x4 = add(x, scale(k3, dt));
                double[] k4 = model.rates(BioreactorState.fromStateVector(t + dt, x4, hiddenCount).withVolume(V), params);

                for (int i = 0; i < x.length; i++) {
                    x[i] = x[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
                    // Clip all concentration fields (not Xv/Xd which are handled by model)
                    if (x[i] < 0 && i >= 2) x[i] = 0;
                }
            }

            // 2. Apply events at the end of step k (after RK4, matching Python)
            //   ORDER: volume reset first, then feed boluses (matching Python)
            for (var ev : stepEvents[k]) {
                if (ev.type() == Simulator.Event.EventType.SAMPLE_VOLUME) {
                    if (ev.measuredVolume() != null) V = ev.measuredVolume();
                }
            }
            for (var ev : stepEvents[k]) {
                if (ev.type() != Simulator.Event.EventType.FEED_BOLUS) continue;
                Map<String, Double> conc = ev.feedConcentrations();
                if (conc == null && ev.feedName() != null) {
                    conc = model.feedStreamConcentrations(ev.feedName(), params);
                }
                if (conc != null && !conc.isEmpty()) {
                    double C_feed = conc.getOrDefault("substrate", 0.0);
                    double C_N = conc.getOrDefault("hidden_0", 0.0);
                    double Vnew = V + ev.dV();
                    double ratio = V / Vnew;
                    double fRatio = ev.dV() / Vnew;

                    // Save pre-dilution hidden states for undiluted indices
                    double[] savedHidden = null;
                    if (x.length > BioreactorState.CORE_STATE_SIZE && !undilutedSet.isEmpty()) {
                        savedHidden = new double[x.length - BioreactorState.CORE_STATE_SIZE];
                        System.arraycopy(x, BioreactorState.CORE_STATE_SIZE, savedHidden, 0, savedHidden.length);
                    }

                    // Mass balance for core states
                    x[0] = x[0] * ratio;  // Xv diluted
                    x[1] = x[1] * ratio;  // Xd diluted
                    x[2] = x[2] * ratio + C_feed * fRatio;  // S
                    x[3] = x[3] * ratio;  // P diluted
                    x[4] = x[4] * ratio;  // B diluted

                    // Hidden states: dilute unless undiluted
                    if (x.length > BioreactorState.CORE_STATE_SIZE) {
                        for (int hi = 0; hi < x.length - BioreactorState.CORE_STATE_SIZE; hi++) {
                            int idx = BioreactorState.CORE_STATE_SIZE + hi;
                            if (undilutedSet.contains(hi)) {
                                x[idx] = savedHidden[hi];  // restore pre-dilution value
                            } else {
                                x[idx] = x[idx] * ratio;  // diluted
                            }
                        }
                        // Feed B carries hidden nutrient N
                        x[BioreactorState.CORE_STATE_SIZE] += C_N * fRatio;
                    }
                    V = Vnew;
                }
            }

            if ((k + 1) * dt <= tEnd) {
                times.add((k + 1) * dt);
                states.add(BioreactorState.fromStateVector((k + 1) * dt, x, hiddenCount).withVolume(V));
            }
        }

        return new SimResult(times, states);
    }

    /** Convert arrays to BioreactorState. */
    public BioreactorState stateAt(List<Double> times, List<BioreactorState> states, double t) {
        int idx = 0;
        for (int i = 0; i < times.size(); i++) {
            if (times.get(i) >= t) { idx = i; break; }
            idx = i;
        }
        if (idx >= times.size() - 1) return states.getLast();

        double t0 = times.get(idx), t1 = times.get(idx + 1);
        if (t1 - t0 < 1e-10) return states.get(idx);
        double frac = (t - t0) / (t1 - t0);
        BioreactorState s0 = states.get(idx), s1 = states.get(idx + 1);
        double[] h = new double[s0.hidden().length];
        for (int i = 0; i < h.length; i++) h[i] = s0.hidden(i) + frac * (s1.hidden(i) - s0.hidden(i));
        return new BioreactorState(t,
            s0.xv() + frac * (s1.xv() - s0.xv()),
            s0.xd() + frac * (s1.xd() - s0.xd()),
            s0.substrate() + frac * (s1.substrate() - s0.substrate()),
            s0.product() + frac * (s1.product() - s0.product()),
            s0.byproduct() + frac * (s1.byproduct() - s0.byproduct()),
            s0.dissolvedO2(), s0.volume(), s0.temperature(), s0.pH(), h);
    }



    private static double[] add(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
        return r;
    }

    private static double[] scale(double[] a, double s) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] * s;
        return r;
    }
}

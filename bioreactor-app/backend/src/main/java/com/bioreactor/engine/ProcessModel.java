package com.bioreactor.engine;

import java.util.List;
import java.util.Map;

/**
 * Interface for a mechanistic process model.
 *
 * Ported from Python: simulation_engine.BioreactorODE and the specific rate functions.
 */
public interface ProcessModel {

    /** Parameters that can be fitted, with start values and bounds. */
    List<ModelParameter> parameters();

    /**
     * Rates of change for each state variable, given the current state and parameters.
     * Returns a double[] of length STATE_SIZE-1 (no time derivative).
     */
    double[] rates(BioreactorState state, Map<String, Double> params);

    /**
     * Concentration of each state variable in each feed stream.
     * Keys are feed names (e.g. "substrate", "byproduct"). Omitted keys = 0.
     */
    Map<String, Double> feedConcentrations(Map<String, Double> params);
}

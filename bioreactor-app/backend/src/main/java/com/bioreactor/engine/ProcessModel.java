package com.bioreactor.engine;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Interface for a mechanistic process model.
 *
 * Ported from Python: simulation_engine.BioreactorODE and the specific rate functions.
 */
public interface ProcessModel {

    /** Parameters that can be fitted, with start values and bounds. */
    List<ModelParameter> parameters();

    /** Number of hidden/internal state variables (beyond the 9 core states). Default 0. */
    default int hiddenStateCount() { return 0; }

    /** Indices of hidden states NOT diluted by feeds (e.g. damage history). Default empty. */
    default java.util.Set<Integer> undilutedHiddenIndices() { return java.util.Collections.emptySet(); }

    /**
     * Rates of change for each state variable, given the current state and parameters.
     * Returns a double[] of length CORE_STATE_SIZE + hiddenStateCount().
     */
    double[] rates(BioreactorState state, Map<String, Double> params);

    /**
     * Concentration of each state variable in each feed stream.
     * Keys are feed names (e.g. "substrate", "byproduct"). Omitted keys = 0.
     */
    Map<String, Double> feedConcentrations(Map<String, Double> params);

    /**
     * Per-feed-stream concentration lookup. Default delegates to feedConcentrations().
     * @param feedName e.g. "feed_b", "feed_c", "substrate"
     * @return map of state variable name to concentration in the feed (e.g. {"substrate": 50.0})
     */
    default Map<String, Double> feedStreamConcentrations(String feedName, Map<String, Double> params) {
        Map<String, Double> all = feedConcentrations(params);
        Double c = all.get(feedName);
        if (c != null && c > 0) {
            return Map.of("substrate", c);
        }
        // Check for "feed_b" → "feedB_S" style mappings in older models
        String paramKey = switch (feedName) {
            case "feed_b" -> "feedB_S";
            case "feed_c" -> "feedC_S";
            case "feed_d" -> "feedD_S";
            default -> null;
        };
        if (paramKey != null && params.containsKey(paramKey)) {
            return Map.of("substrate", params.get(paramKey));
        }
        return Map.of();
    }
}

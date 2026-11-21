package com.bioreactor.controller;

import com.bioreactor.engine.*;
import com.bioreactor.engine.ConfigurableModel.*;
import com.bioreactor.service.OrganismConfigLoader;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

/**
 * REST API for the bioreactor process application.
 *
 * Endpoints:
 *   GET  /api/v1/organisms                    - list available organisms
 *   GET  /api/v1/organisms/{id}               - organism config with parameters
 *   POST /api/v1/simulate                     - run a simulation for a given organism
 *
 * This is the production API for the manufacturing monitoring system,
 * not the DTB challenge-specific prediction endpoint.
 */
@RestController
@RequestMapping("/api/v1")
public class OrganismController {

    private final OrganismConfigLoader configLoader;

    public OrganismController(OrganismConfigLoader configLoader) {
        this.configLoader = configLoader;
    }

    /** List all available organisms. */
    @GetMapping("/organisms")
    public ResponseEntity<List<OrganismSummary>> listOrganisms() {
        List<OrganismSummary> result = new ArrayList<>();
        for (String id : configLoader.listOrganisms()) {
            OrganismConfig c = configLoader.load(id);
            result.add(new OrganismSummary(
                c.id(), c.name(),
                c.growthModel(), c.muMax(), c.kS(), c.yXS(),
                c.substrates().size(),
                c.products().size() + c.byproducts().size()));
        }
        return ResponseEntity.ok(result);
    }

    /** Get full organism configuration with parameters. */
    @GetMapping("/organisms/{id}")
    public ResponseEntity<OrganismDetail> getOrganism(@PathVariable String id) {
        OrganismConfig c = configLoader.load(id);
        ConfigurableModel model = new ConfigurableModel(c);

        List<ModelParameter> params = model.parameters();
        Map<String, Object> paramMap = new LinkedHashMap<>();
        for (ModelParameter mp : params) {
            paramMap.put(mp.name(), Map.of(
                "start", mp.startValue(),
                "min",   mp.lowerBound(),
                "max",   mp.upperBound(),
                "unit",  mp.unit(),
                "description", mp.description()));
        }

        return ResponseEntity.ok(new OrganismDetail(
            c.id(), c.name(), c.growthModel(),
            c.muMax(), c.kS(), c.yXS(), c.mS(),
            c.tMin(), c.tOpt(), c.tMax(),
            c.pHMin(), c.pHOpt(), c.pHMax(),
            c.substrates(), c.products(), c.byproducts(),
            paramMap));
    }

    /** Run a simulation with custom parameters. */
    @PostMapping("/simulate")
    public ResponseEntity<SimulationResponse> simulate(@RequestBody SimulationRequest req) {
        OrganismConfig config = configLoader.load(req.organismId);
        ConfigurableModel model = new ConfigurableModel(config);

        // Merge default params with overrides from request
        Map<String, Double> params = new HashMap<>();
        for (ModelParameter mp : model.parameters()) {
            params.put(mp.name(), mp.startValue());
        }
        if (req.parameterOverrides != null) {
            params.putAll(req.parameterOverrides);
        }

        // Initial state
        int dim = BioreactorState.CORE_STATE_SIZE + model.hiddenStateCount();
        double[] initialState = new double[dim];
        initialState[0] = req.initialXv > 0 ? req.initialXv : 0.5;  // Xv
        initialState[2] = req.initialSubstrate > 0 ? req.initialSubstrate : 10.0; // S0
        // DO
        int idxDO = 2 + config.substrates().size() + config.products().size() + config.byproducts().size();
        initialState[idxDO] = 0.007; // saturation
        // Temperature & pH
        initialState[idxDO + 2] = req.temperature > 0 ? req.temperature : config.tOpt();
        initialState[idxDO + 3] = req.pHValue > 0 ? req.pHValue : config.pHOpt();

        BioreactorState init = new BioreactorState(0.0, initialState[0], initialState[1],
            initialState[2], 0.0, 0.0, initialState[idxDO], req.volume > 0 ? req.volume : 1.0,
            initialState[idxDO + 2], initialState[idxDO + 3]);

        // Run simulation
        RK4Simulator sim = new RK4Simulator(model, 0.01);
        double tEnd = req.tEnd > 0 ? req.tEnd : 24.0;
        var result = sim.simulate(init, params, List.of(), tEnd);

        // Build response
        List<SimulationPoint> points = new ArrayList<>();
        for (int i = 0; i < result.times().size(); i++) {
            BioreactorState s = result.states().get(i);
            points.add(new SimulationPoint(
                result.times().get(i),
                s.xv(), s.xd(), s.xv() + s.xd(),
                s.substrate(), s.byproduct(),
                s.volume(), s.temperature(), s.pH()));
        }

        return ResponseEntity.ok(new SimulationResponse(
            req.organismId, req.initialSubstrate, req.tEnd, points));
    }

    // ── Request / response records ──

    public record SimulationRequest(
        String organismId,
        double initialXv, double initialSubstrate, double volume,
        double temperature, double pHValue, double tEnd,
        Map<String, Double> parameterOverrides) {}

    public record OrganismSummary(
        String id, String name, String growthModel,
        double muMax, double kS, double yXS,
        int substrateCount, int productCount) {}

    public record OrganismDetail(
        String id, String name, String growthModel,
        double muMax, double kS, double yXS, double mS,
        double tMin, double tOpt, double tMax,
        double pHMin, double pHOpt, double pHMax,
        List<SubstrateConfig> substrates,
        List<ProductConfig> products,
        List<ProductConfig> byproducts,
        Map<String, Object> parameters) {}

    public record SimulationPoint(
        double time,
        double xv, double xd, double tcd,
        double substrate, double byproduct,
        double volume, double temperature, double pH) {}

    public record SimulationResponse(
        String organismId, double initialSubstrate, double tEnd,
        List<SimulationPoint> points) {}
}

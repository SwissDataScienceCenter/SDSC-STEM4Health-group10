package com.bioreactor.service;

import com.bioreactor.engine.*;
import com.bioreactor.engine.ConfigurableModel.OrganismConfig;
import com.bioreactor.model.*;
import com.bioreactor.repository.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;

/**
 * Pure Java simulation service — runs ConfigurableModel via RK4Simulator.
 * Persists results to simulation_runs and simulation_results tables.
 *
 * Replaces the deprecated Python ProcessBuilder bridge.
 */
@Service
public class SimulationService {

    private final SimulationRunRepository simulationRunRepository;
    private final SimulationResultRepository simulationResultRepository;
    private final OrganismRepository organismRepository;
    private final OrganismConfigLoader configLoader;

    public SimulationService(SimulationRunRepository simulationRunRepository,
                             SimulationResultRepository simulationResultRepository,
                             OrganismRepository organismRepository,
                             OrganismConfigLoader configLoader) {
        this.simulationRunRepository = simulationRunRepository;
        this.simulationResultRepository = simulationResultRepository;
        this.organismRepository = organismRepository;
        this.configLoader = configLoader;
    }

    /** List organisms from the database. */
    public List<Map<String, Object>> listOrganisms() {
        List<Organism> orgs = organismRepository.findAll();
        List<Map<String, Object>> result = new ArrayList<>();
        for (Organism o : orgs) {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("id", o.getId());
            map.put("name", o.getName());
            map.put("type", o.getOrganismType());
            map.put("description", o.getDescription());
            result.add(map);
        }
        return result;
    }

    /** Create a new simulation run entity (pending). */
    @Transactional
    public SimulationRun createSimulation(String organismId, double xV, double substrate,
                                          double volume, double temperature, double pH,
                                          double tEnd, String name) {
        SimulationRun run = new SimulationRun();
        run.setName(name != null ? name : "Simulation: " + organismId);
        run.settEnd(tEnd);
        run.setStatus("pending");
        return simulationRunRepository.save(run);
    }

    /**
     * Run a simulation using the pure Java engine (RK4Simulator + ConfigurableModel).
     * Persists results as SimulationResult rows.
     */
    @Transactional
    public Map<String, Object> runSimulation(SimulationRun run, String organismId,
                                              double xV, double s0, double v, double temp, double ph) {
        run.setStatus("running");
        run.setStartedAt(LocalDateTime.now());
        simulationRunRepository.save(run);

        Map<String, Object> results = new LinkedHashMap<>();

        try {
            OrganismConfig config = configLoader.load(organismId);
            ConfigurableModel model = new ConfigurableModel(config);

            Map<String, Double> params = new HashMap<>();
            for (ModelParameter mp : model.parameters()) params.put(mp.name(), mp.startValue());

            BioreactorState initialState = new BioreactorState(
                0.0, xV, 0.0, s0, 0.0, 0.0,
                0.007, v > 0 ? v : 1.0, temp > 0 ? temp : config.tOpt(),
                ph > 0 ? ph : config.pHOpt());

            RK4Simulator sim = new RK4Simulator(model, 0.01);
            double tEnd = run.gettEnd() > 0 ? run.gettEnd() : 24.0;
            var simResult = sim.simulate(initialState, params, List.of(), tEnd);

            // Persist results at each output step
            List<SimulationResult> savedResults = new ArrayList<>();
            for (int i = 0; i < simResult.times().size(); i++) {
                BioreactorState s = simResult.states().get(i);
                double tcd = s.xv() + s.xd();
                double viability = tcd > 0 ? s.xv() / tcd : 1.0;

                SimulationResult sr = new SimulationResult();
                sr.setSimulationId(run.getId());
                sr.setTime(simResult.times().get(i));
                sr.setxV(s.xv());
                sr.setxD(s.xd());
                sr.setSubstrate(s.substrate());
                sr.setProduct(s.product());
                sr.setByproduct(s.byproduct());
                sr.setViability(viability);
                sr.setTemperature(s.temperature());
                sr.setpH(s.pH());
                sr.setDissolvedO2(s.dissolvedO2());
                sr.setVolume(s.volume());
                savedResults.add(simulationResultRepository.save(sr));
            }

            run.setStatus("completed");
            results.put("status", "completed");
            results.put("message", "Simulation completed in pure Java");
            results.put("simulation_id", run.getId());
            results.put("point_count", savedResults.size());
            results.put("organism", organismId);

        } catch (Exception e) {
            results.put("status", "error");
            results.put("message", e.getMessage());
            run.setStatus("failed");
        }

        run.setCompletedAt(LocalDateTime.now());
        simulationRunRepository.save(run);

        return results;
    }

    /** Retrieve simulation results grouped by time series. */
    public List<Map<String, Object>> getSimulationResults(Integer simulationId) {
        List<SimulationResult> rows = simulationResultRepository.findBySimulationIdOrderByTimeAsc(simulationId);
        List<Map<String, Object>> results = new ArrayList<>();

        List<Double> time = new ArrayList<>();
        List<Double> xv = new ArrayList<>();
        List<Double> xd = new ArrayList<>();
        List<Double> substrate = new ArrayList<>();
        List<Double> product = new ArrayList<>();
        List<Double> byproduct = new ArrayList<>();
        List<Double> viability = new ArrayList<>();
        List<Double> temperature = new ArrayList<>();
        List<Double> ph = new ArrayList<>();
        List<Double> dissolvedO2 = new ArrayList<>();
        List<Double> volume = new ArrayList<>();

        for (SimulationResult r : rows) {
            time.add(r.getTime());
            xv.add(r.getxV());
            xd.add(r.getxD());
            substrate.add(r.getSubstrate());
            product.add(r.getProduct());
            byproduct.add(r.getByproduct());
            viability.add(r.getViability());
            temperature.add(r.getTemperature());
            ph.add(r.getpH());
            dissolvedO2.add(r.getDissolvedO2());
            volume.add(r.getVolume());
        }

        Map<String, Object> series = new LinkedHashMap<>();
        series.put("time", time);
        series.put("X_v", xv);
        series.put("X_d", xd);
        series.put("substrate", substrate);
        series.put("product", product);
        series.put("byproduct", byproduct);
        series.put("viability", viability);
        series.put("temperature", temperature);
        series.put("pH", ph);
        series.put("dissolvedO2", dissolvedO2);
        series.put("volume", volume);

        results.add(series);
        return results;
    }

    public List<SimulationRun> getSimulations() {
        return simulationRunRepository.findAll();
    }

    public Optional<SimulationRun> getSimulation(Integer id) {
        return simulationRunRepository.findById(id);
    }
}

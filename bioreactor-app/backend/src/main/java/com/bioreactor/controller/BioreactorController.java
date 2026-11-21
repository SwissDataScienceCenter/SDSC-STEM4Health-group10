package com.bioreactor.controller;

import com.bioreactor.model.SimulationRun;
import com.bioreactor.service.SimulationService;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class BioreactorController {

    private final SimulationService simulationService;

    public BioreactorController(SimulationService simulationService) {
        this.simulationService = simulationService;
    }

    @GetMapping("/organisms")
    public List<Map<String, Object>> listOrganisms() {
        return simulationService.listOrganisms();
    }

    @PostMapping("/simulate")
    public Map<String, Object> runSimulation(@RequestBody Map<String, Object> request) {
        String organismId = (String) request.getOrDefault("organism_id", "ecoli_k12");
        double xV = getDouble(request, "X_v", 1.0);
        double s = getDouble(request, "S", 5.0);
        double v = getDouble(request, "V", 10.0);
        double temp = getDouble(request, "temperature", 37.0);
        double pH = getDouble(request, "pH", 7.0);
        double tEnd = getDouble(request, "t_end", 24.0);
        String name = (String) request.getOrDefault("name", null);

        SimulationRun run = simulationService.createSimulation(organismId, xV, s, v, temp, pH, tEnd, name);
        return simulationService.runSimulation(run, organismId, xV, s, v, temp, pH);
    }

    @GetMapping("/simulations")
    public List<Map<String, Object>> listSimulations() {
        List<SimulationRun> sims = simulationService.getSimulations();
        List<Map<String, Object>> result = new ArrayList<>();
        for (SimulationRun s : sims) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", s.getId());
            m.put("name", s.getName());
            m.put("status", s.getStatus());
            m.put("t_end", s.gettEnd());
            m.put("created_at", s.getCreatedAt() != null ? s.getCreatedAt().toString() : null);
            result.add(m);
        }
        return result;
    }

    @GetMapping("/simulations/{id}/results")
    public List<Map<String, Object>> getSimulationResults(@PathVariable Integer id) {
        return simulationService.getSimulationResults(id);
    }

    @GetMapping("/simulations/{id}")
    public Map<String, Object> getSimulation(@PathVariable Integer id) {
        Optional<SimulationRun> sim = simulationService.getSimulation(id);
        Map<String, Object> result = new LinkedHashMap<>();
        if (sim.isPresent()) {
            SimulationRun s = sim.get();
            result.put("id", s.getId());
            result.put("name", s.getName());
            result.put("status", s.getStatus());
            result.put("t_end", s.gettEnd());
            return result;
        }
        result.put("error", "Not found");
        return result;
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "ok", "service", "bioreactor-api");
    }

    private double getDouble(Map<String, Object> map, String key, double defaultValue) {
        Object val = map.get(key);
        if (val instanceof Number n) return n.doubleValue();
        if (val instanceof String s) {
            try { return Double.parseDouble(s); } catch (NumberFormatException e) {}
        }
        return defaultValue;
    }
}

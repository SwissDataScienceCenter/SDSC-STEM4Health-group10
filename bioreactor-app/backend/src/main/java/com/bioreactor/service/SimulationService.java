package com.bioreactor.service;

import com.bioreactor.repository.*;
import com.bioreactor.model.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.*;
import java.time.LocalDateTime;
import java.util.*;

@Service
public class SimulationService {

    private final SimulationRunRepository simulationRunRepository;
    private final SimulationResultRepository simulationResultRepository;
    private final OrganismRepository organismRepository;
    
    @Value("${bioreactor.python.path}")
    private String pythonPath;
    
    @Value("${bioreactor.python.script}")
    private String pythonScript;

    public SimulationService(SimulationRunRepository simulationRunRepository,
                             SimulationResultRepository simulationResultRepository,
                             OrganismRepository organismRepository) {
        this.simulationRunRepository = simulationRunRepository;
        this.simulationResultRepository = simulationResultRepository;
        this.organismRepository = organismRepository;
    }

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

    @Transactional
    public SimulationRun createSimulation(String organismId, double xV, double substrate,
                                          double volume, double temperature, double pH,
                                          double tEnd, String name) {
        SimulationRun run = new SimulationRun();
        run.setName(name != null ? name : "Simulation: " + organismId);
        run.settEnd(tEnd);
        run.setStatus("pending");
        run.setConfigId(null);
        return simulationRunRepository.save(run);
    }

    @Transactional
    public Map<String, Object> runSimulation(SimulationRun run, String organismId,
                                              double xV, double s, double v, double temp, double ph) {
        run.setStatus("running");
        run.setStartedAt(LocalDateTime.now());
        simulationRunRepository.save(run);

        Map<String, Object> results = new LinkedHashMap<>();
        
        try {
            String inputJson = String.format(Locale.US,
                "{\"organism_id\":\"%s\",\"X_v\":%f,\"S\":%f,\"V\":%f,\"temperature\":%f,\"pH\":%f,\"t_end\":%f,\"simulation_id\":%d}",
                organismId, xV, s, v, temp, ph, run.gettEnd(), run.getId());
            
            ProcessBuilder pb = new ProcessBuilder(pythonPath, pythonScript);
            pb.redirectErrorStream(true);
            
            Map<String, String> env = pb.environment();
            env.put("PYTHONPATH", new File(pythonScript).getParent());
            
            Process process = pb.start();
            
            // Send input
            try (OutputStream os = process.getOutputStream()) {
                os.write(inputJson.getBytes());
                os.flush();
            }
            
            // Read output
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("{")) {
                        output.append(line);
                    }
                }
            }
            
            int exitCode = process.waitFor();
            
            if (exitCode == 0 && output.length() > 0) {
                results.put("status", "completed");
                results.put("message", "Simulation completed");
                results.put("simulation_id", run.getId());
                results.put("raw", output.toString());
                
                run.setStatus("completed");
            } else {
                results.put("status", "failed");
                results.put("message", "Python exit code: " + exitCode);
                run.setStatus("failed");
            }
            
        } catch (Exception e) {
            results.put("status", "error");
            results.put("message", e.getMessage());
            run.setStatus("failed");
        }
        
        run.setCompletedAt(LocalDateTime.now());
        simulationRunRepository.save(run);
        
        return results;
    }

    public List<Map<String, Object>> getSimulationResults(Integer simulationId) {
        List<SimulationResult> rows = simulationResultRepository.findBySimulationIdOrderByTimeAsc(simulationId);
        List<Map<String, Object>> results = new ArrayList<>();
        
        // Group by time series
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

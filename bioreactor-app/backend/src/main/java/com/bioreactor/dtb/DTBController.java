package com.bioreactor.dtb;

import com.bioreactor.engine.Fitter;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;

import java.io.*;
import java.nio.file.Files;
import java.util.*;

@RestController
@RequestMapping("/api/dtb")
public class DTBController {

    private final DTBPredictionService service;

    public DTBController(DTBPredictionService service) {
        this.service = service;
    }

    /** DTO for fit response. */
    public record FitResponse(
        Map<String, Double> fittedParams,
        boolean converged,
        double finalCost,
        Map<String, Double> mapePerVariable,
        int iterations
    ) {}

    @PostMapping("/fit")
    public ResponseEntity<?> fit() {
        try {
            Fitter.FitResult r = service.fitOnTrainingData();
            return ResponseEntity.ok(new FitResponse(
                r.fittedParams(), r.converged(), r.finalCost(),
                r.mapePerVariable(), r.iterations()));
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                .body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/predict")
    public ResponseEntity<?> predict(
        @RequestParam(defaultValue = "false") boolean download,
        @RequestParam Map<String, String> allParams
    ) throws Exception {

        // Use fitted defaults if no params provided
        Map<String, Double> params = new HashMap<>();
        if (allParams.isEmpty()) {
            // Default DTB consensus params from Python fit
            params.put("mu_max", 0.73);
            params.put("K_s", 0.009);
            params.put("Y_XS", 0.55);
            params.put("m_s", 0.03);
            params.put("alpha", 0.04);
            params.put("beta", 0.0001);
            params.put("k_d", 0.005);
            params.put("k_lysis", 0.001);
            params.put("feedB_S", 500.0);
        } else {
            // Also do a fit to compute params
            Fitter.FitResult r = service.fitOnTrainingData();
            params = r.fittedParams();
        }

        var outputs = service.generatePredictions(params);
        File csv = service.writePredictionCSV(outputs);

        if (download) {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.parseMediaType("text/csv"));
            headers.setContentDisposition(ContentDisposition.attachment()
                .filename("dtb_predictions.csv").build());
            byte[] bytes = Files.readAllBytes(csv.toPath());
            csv.delete();
            return new ResponseEntity<>(bytes, headers, HttpStatus.OK);
        }

        // JSON summary: first few rows per batch
        List<Map<String, Object>> summary = new ArrayList<>();
        for (var out : outputs) {
            Map<String, Object> batch = new LinkedHashMap<>();
            batch.put("batch_id", out.batchId());
            List<Map<String, Double>> rows = new ArrayList<>();
            for (var row : out.rows()) {
                rows.add(Map.of("t", row.time(), "vcd", row.vcd(),
                    "tcd", row.tcd(), "substrate", row.substrate(),
                    "byproduct", row.byproduct()));
            }
            batch.put("rows", rows);
            summary.add(batch);
        }

        Map<String, Object> response = Map.of(
            "parameters", params,
            "batches", summary
        );

        csv.delete();
        return ResponseEntity.ok(response);
    }
}

package com.bioreactor.controller;

import com.bioreactor.service.ProcessRunner;
import com.bioreactor.service.ProcessRunner.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

/**
 * REST API for manufacturing process monitoring and analysis.
 *
 * Endpoints:
 *   GET  /api/v2/runs                        - list manufacturing runs
 *   GET  /api/v2/runs/{runId}                - run metadata
 *   GET  /api/v2/runs/{runId}/readings       - time-series readings
 *   GET  /api/v2/runs/{runId}/feed-events    - feed/sample events
 *   POST /api/v2/runs/{runId}/analyze        - run full analysis (simulate + compare)
 */
@RestController
@RequestMapping("/api/v2")
public class ProcessController {

    private final ProcessRunner runner;

    public ProcessController(ProcessRunner runner) {
        this.runner = runner;
    }

    /** List all manufacturing runs in the system. */
    @GetMapping("/runs")
    public ResponseEntity<List<Map<String, Object>>> listRuns() {
        List<Map<String, Object>> result = runner.listRuns();
        return ResponseEntity.ok(result);
    }

    /** Get run metadata. */
    @GetMapping("/runs/{runId}")
    public ResponseEntity<Map<String, Object>> getRun(@PathVariable String runId) {
        return ResponseEntity.ok(runner.loadRunMeta(runId));
    }

    /** Get time-series readings for a run. */
    @GetMapping("/runs/{runId}/readings")
    public ResponseEntity<List<TimePoint>> getReadings(@PathVariable String runId) {
        return ResponseEntity.ok(runner.loadReadings(runId));
    }

    /** Get feed events for a run. */
    @GetMapping("/runs/{runId}/feed-events")
    public ResponseEntity<List<FeedEvent>> getFeedEvents(@PathVariable String runId) {
        return ResponseEntity.ok(runner.loadFeedEvents(runId));
    }

    /** Full analysis: simulate and compare to measurements. */
    @PostMapping("/runs/{runId}/analyze")
    public ResponseEntity<ProcessReport> analyzeRun(
        @PathVariable String runId,
        @RequestBody Map<String, String> body) {
        String organismId = body.getOrDefault("organism_id", "ecoli_k12");
        ProcessReport report = runner.analyzeRun(runId, organismId);
        return ResponseEntity.ok(report);
    }

    /** Trending analysis across training batches. */
    @GetMapping("/trends")
    public ResponseEntity<TrendReport> trendAnalysis(
        @RequestParam String processId,
        @RequestParam(defaultValue = "peak_vcd") String parameter) {
        TrendReport report = runner.trendAnalysis(processId, parameter);
        return ResponseEntity.ok(report);
    }
}

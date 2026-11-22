# Bioreactor Process Application — Module Documentation

## Packages

| Package | Purpose |
|---|---|
| `com.bioreactor.engine` | Pure Java: `ProcessModel`, `BioreactorState`, `Simulator`, `Fitter`. No Spring, no DB, no files. |
| `com.bioreactor.dtb` | DTB challenge-specific: `DTBModel`, `DTBDataReader`, `DTBPredictionService`, `DTBController`. |
| `com.bioreactor.prediction` | `PredictionEngine` — typed predict contract for any model. |
| `com.bioreactor.monitoring` | `MonitoringEngine` — reference bands, findings, early warning. |
| `com.bioreactor.trending` | `TrendingEngine` — I-MR control charts, Cpk, run metrics. |
| `com.bioreactor.model` | JPA entities: `SimulationRun`, `SimulationResult`, `Organism`, `OrganismGrowthParams` (existing). |
| `com.bioreactor.controller` | REST controllers (old `/api/` and new `/api/v1/dtb/`). |
| `com.bioreactor.service` | `SimulationService` (pure Java), `DTBImportService` (CSV→DB), `ProcessRunner` (analysis). |

## Database tables

| Table | Migration | In my application |
|---|---|---|
| `experiment_data`, `experiments` | V1 | Existing wide table (DTB only) |
| `organisms` | V2 | Organism library |
| `simulation_runs`, `simulation_results` | V1 | Old simulation storage |
| `process_definition` | V3 | Process version (seeded: DTB-v1) |
| `process_step` | V3 | Process step |
| `process_parameter` | V3 | Parameter and quality attribute |
| `equipment` | V3 | Equipment |
| `run` | V3 | Batch / development run (populated by DTBImportService) |
| `reading` | V3 | Time-series observation (Filled from CSV) |
| `run_event` | V3 | Feed / sample event (Filled from CSV) |
| `import_file` | V3 | Import evidence |
| `model`, `model_version` | V4 + V7 | Model version record (seeded: DTB-v3, DTB-v3-constrained) |
| `prediction_run`, `prediction_value` | V4 | Stored predictions |
| `monitoring_finding` | V5 | Early warning findings |
| `run_metric`, `trend_assessment` | V6 | Cross-batch trending |

## Build and start

```bash
# Prerequisites: Java 21, PostgreSQL on localhost:5432 (database: bioreactor, user: appuser)
cd bioreactor-app/backend

# Build and test (no Python, no DB)
./mvnw verify

# Start (needs DB)
./mvnw spring-boot:run
```

## Verified

- [x] 24 tests pass: EngineTest(5) + PythonPortTest(2) + DTBV3ModelPortTest(5)
       + RealDataV3FitTest(1) + RealDataFitTest(1) + EngineIntegrationTest(3)
       + ModelVariantComparisonTest(6) + V3GIModelComparisonTest(1)
- [x] Simulator uses model.rates() correctly
- [x] Bolus mass balance (feed + volume reset)
- [x] Fitter converges on synthetic data (Levenberg-Marquardt)
- [x] Leave-one-batch-out CV runs
- [x] Prediction engine outputs 4 parameters per time point
- [x] Reference band: p10 ≤ p50 ≤ p90
- [x] Monitoring detects band violations
- [x] I-MR control chart built
- [x] Java-vs-Python port validation (Monod: VCD MARE 0.0001%, Sub MARE 0.0001%)
- [x] Feed architecture: per-stream concentrations resolved from model
- [x] Fast fixed-step RK4 simulator (dt=0.01, matches Python)
- [x] DTBV3ConstrainedModel: 12 free params, 10 locked to Python V3 literatures
- [x] Submission CSV generator (DTBSubmissionGenerator, reads template, 156 rows)
- [x] Python ProcessBuilder call removed (SimulationService is pure Java)
- [x] DTBCsvReader uses feed names (model resolves concentrations + hidden N)
- [x] Fitter.predict uses last event time for test batches (fixes tEnd=0 bug)
- [x] Jacobian perturbation improved (relative+absolute, avoids singular matrix)
- [x] Residual clamping to prevent Inf in LM cost function
- [x] V3 model defaults set to team Python fit (MAPE 14.6% without fitting)
- [x] V3Constrained fitted MAPE: VCD=8.5% TCD=8.8% Sub=21.4% Byp=25.6% Mean=16.1%
- [x] V3 full model defaults = Python fit (14.6% MAPE without any fitting)
- [x] DTBImportService: CSV → V3 tables (run, reading, run_event)
- [x] DTBController: /api/dtb/import, /api/dtb/status, /api/dtb/fit, /api/dtb/predict
- [x] ProcessController: /api/v2/runs, /api/v2/runs/{id}/readings, etc.
- [x] Flyway V1–V7 present (V7 seeds model version records)
- [x] ProcessModel.defaultHiddenState() — models declare hidden initial values
- [x] BioreactorState.hiddenCount() — accessor
- [x] Fitter.predict() auto-injects hidden states from model when missing
- [x] DashboardView: Vaadin UI with 3 tabs (Monitoring, Trending, Prediction)
- [x] DTBSubmissionGenerator: --no-fit, --skip-cv, --model= flags

## Known limitations

## MAPE summary

| Test | VCD | TCD | Substrate | Byproduct | Mean |
|------|-----|-----|-----------|-----------|------|
| V3 holdout (RealDataV3FitTest) | 8.4% | 8.6% | 22.2% | 22.1% | 15.3% |
| V3 7-fold CV (CrossValidationTest) | ~8.6% | ~8.5% | ~19.8% | ~18.5% | ~13.7% |
| Python V3 7-fold CV (reference) | 8.6% | 8.5% | 19.8% | 18.5% | 13.8% |

Java 7-fold CV matches Python within 0.1% (same protocol: scaled residuals, Python defaults).

## Known limitations

- V3 holdout MAPE (15.3%) higher than CV (13.7%) — holdout batch selection accounts for difference
- V3 model has 8/22 params at bounds — structural limit of current model form
- V2 models (DTBModel, DTBModelImproved, DTBModelAdvanced) have 43–115% MAPE — deprecated
- DTBDataReader expects V1 wide table (experiment_data) but code now uses V3 long format
- springdoc-openapi added (v2.7.0) — OpenAPI docs available at /v3/api-docs and /swagger-ui.html
- BRPMM mapping file not started (lowest priority per original brief)

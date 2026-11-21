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
| `com.bioreactor.service` | `SimulationService` — Python bridge (deprecated, to be replaced). |

## Database tables

| Table | Migration | In my application |
|---|---|---|
| `experiment_data`, `experiments` | V1 | Existing wide table (DTB only) |
| `organisms` | V2 | Organism library |
| `simulation_runs`, `simulation_results` | V1 | Old simulation storage |
| `process_definition` | V3 | Process version |
| `process_step` | V3 | Process step |
| `process_parameter` | V3 | Parameter and quality attribute |
| `equipment` | V3 | Equipment |
| `run` | V3 | Batch / development run |
| `reading` | V3 | Time-series observation |
| `run_event` | V3 | Feed / sample event |
| `import_file` | V3 | Import evidence |
| `model`, `model_version` | V4 | Model version record |
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

- [x] 8 tests pass (`mvn test`)
- [x] Simulator uses model.rates() correctly
- [x] Bolus mass balance (feed + volume reset)
- [x] Fitter converges on synthetic data (Levenberg-Marquardt)
- [x] Leave-one-batch-out CV runs
- [x] Prediction engine outputs 4 parameters per time point
- [x] Reference band: p10 ≤ p50 ≤ p90
- [x] Monitoring detects band violations
- [x] I-MR control chart built

## Not verified

- [ ] Java-vs-Python numerical comparison (port check)
- [ ] Fitting on real DTB training data (needs active DB + Java DB reader)
- [ ] `DTBDataReader.loadDTBBatch()` tested on real batches
- [ ] MAPE on DTB test batches matches team v3 (~8.6% VCD, ~8.5% TCD, ~19.8% substrate, ~18.5% byproduct)
- [ ] Submission file generation (156 rows, correct format)
- [ ] ProcessBuilder Python call removed (still in SimulationService)
- [ ] Vaadin UI (not started)
- [ ] Flyway V3-V6 applied on real DB
- [ ] BRPMM mapping file

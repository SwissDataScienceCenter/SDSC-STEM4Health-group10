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
| `com.bioreactor.service` | `SimulationService` — pure Java simulation (Python bridge removed). |

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

- [x] 13 tests pass (`mvn test`), including V3Constrained on real data
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
- [x] DTBV3ConstrainedModel: 12 free params, 10 locked to CHO literature
- [x] Submission CSV generator (DTBSubmissionGenerator, reads template, 156 rows)
- [x] Python ProcessBuilder call removed (SimulationService is pure Java)
- [x] DTBCsvReader uses feed names (model resolves concentrations + hidden N)
- [x] Fitter.predict uses last event time for test batches (fixes tEnd=0 bug)
- [x] Jacobian perturbation improved (relative+absolute, avoids singular matrix)
- [x] Residual clamping to prevent Inf in LM cost function

## In progress

- [ ] MAPE on DTB batches matching team v3 (~8.6% VCD, ~8.5% TCD, ~19.8% sub)
  Current: DTBV3ConstrainedModel CV MAPE ~61% VCD, 55% TCD, 33% Sub, 44% Byp
  The 10 locked literature parameters need tuning for this data set.
  The unfitted V3 model (22 params) has lower holdout MAPE (21%) but does not
  converge when fitted — parameters highly correlated, model may be overparameterized.

## Not yet done

- [ ] DTBDataReader tested on all real batches (needs running DB)
- [ ] Vaadin UI complete (DashboardView exists but may need work)
- [ ] Flyway V3-V6 applied on real DB
- [ ] BRPMM mapping file
- [ ] Full leave-one-batch-out CV on all 49 training batches

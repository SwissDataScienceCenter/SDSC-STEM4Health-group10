# Experiments of 22 Nov 2026 (after the team blend, LB 7.725)

All CV numbers: 7 folds by batch, 6 seeds; choices made on seeds 0-2, checked on fresh seeds 3-5.
MAPE in %, leaderboard (LB) = 12 test batches.

| Step | CV effect | LB | Verdict |
|---|---|---|---|
| Pooled-time SVR + trees for vcd/tcd (`src/model_families.py`), 0.25 team + 0.55 SVR + 0.20 trees | vcd -0.108, tcd -0.114 | **7.686** (tcd 7.178 -> 7.044) | kept |
| Late substrate dose rule (`control_rules.LateDoseRule`): dose size after samples 11/12 reveals S < 0.96 | substrate -0.14 | **7.613** (substrate 8.297 -> 8.003) | kept: `submissions/blend_nonlinear_dose.csv` |
| Mid-run bolus rule re-applied on the final blend | substrate -0.05 | 7.627 (substrate 8.061) | dropped (worse on LB) |
| Non-linear families for substrate / byproduct (LightGBM, RF, extra-trees, kernel ridge, SVR, MLP) | +2.5 to +4 points | - | dropped: feed jumps at fixed times favour per-time models |
| Wave 2 (`src/experiments/wave2_cv.py`): time-weighted kernels, varying-coefficient ridge, residual-stacked SVR, per-time SVR, feed-event features | ensemble gain 0.002 | - | dropped |
| Feed B dose features (`src/feedb_controller.py`) | substrate +0.36 to +1.4 (worse) | - | dropped: Feed B replaces the substrate consumed the day before (corr -0.85 with S ratio), i.e. it reveals S differences, not levels |
| Feed C small-dose counts (pH control?) as ridge features | vcd -0.02 | - | negligible (already in the teammate model as nCs_tot) |
| MAPE-optimal downward calibration exp(-sigma^2), regime-bias correction | worse | - | dropped (cross-fitting on OOF errors is biased against corrections; as features: overfit, 7 controlled batches) |
| Regime indicators (mid-run bolus, late dose classes) as ridge features | substrate +0.4 to +0.9 (worse) | - | dropped |
| Importance weighting for train/test shift | - | - | no detectable shift: train-vs-test classifier AUC 0.43 |

## Ceilings (why the remaining gap is hard)
* **ODE v3 with per-batch parameters fitted to the measurements (oracle)**: vcd 5.97, tcd 5.73, substrate 17.2,
  byproduct 15.3 % (statistical models without measurements: 6.5 / 6.4 / 6.7 / 8.7). The fitted v3 put the
  substrate-feed concentration at its lower bound (1 g/L instead of ~300 g/L by mass balance): its substrate and
  byproduct equations cannot be rescued by per-batch calibration ("controller assimilation" not pursued).
* **Late substrate mass balance with the TRUE previous sample** (S*V change ~ substrate dose, Feed B, IVCD; fitted
  C_S ~ 320, R^2 0.90 / 0.73): S_12 7.0 %, S_13 10.2 % (current models without any measurement: 10.8 / 10.7 %).
  Most of the remaining late-substrate error is batch-to-batch variability of consumption.
* Error budget of the current best (CV): substrate samples 11-13 = 42 % of the substrate error, byproduct
  samples 11-13 = 35 % of the byproduct error.

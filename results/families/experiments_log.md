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

## Hidden base volume (team note) and its optimisation, 22 Nov afternoon
| Step | CV (seeds 0-2 / 3-5) | LB | Verdict |
|---|---|---|---|
| Note's Huber byproduct model with `base_production` (hidden volume, intervals 3-5), 0.9 + 0.1 teammate (`blend_base.csv`) | byproduct 8.67 -> 7.90 / 7.83 | **7.464** (byproduct 8.241 -> 7.644) | kept |
| `blend_base2.csv`: + base_production in our model for substrate (0.75 ours_base + 0.25 teammate) | substrate 6.70 / 6.81 -> 6.48 / 6.58 | pending | superseded by base3 |
| Window / correction search (`src/experiments/base_opt.py`): hidden volume minus an evaporation baseline fitted on the late intervals (hidden ~ -0.020 - 0.031 * dt) | substrate ridge: window 2-5 6.72 / 6.91 vs 3-5 6.93 / 7.12; byproduct: corrected 3-5 7.90 / 7.82 vs 7.92 / 7.84 | - | kept (base_c2345 substrate, base_c345 byproduct) |
| Byproduct Huber settings | alpha 0.3, epsilon 2.0: 7.88 / 7.81 (weight of typical curve 0.2 and no neighbour-day pooling stay best) | - | kept |
| Base features for vcd / tcd (ridge and SVR) | no confirmed gain | - | not used |
| Late-phase hidden volume (intervals 11-13), per-batch baseline, separate intervals | not confirmed on seeds 3-5 | - | not used |
| **`blend_base3.csv`**: substrate 0.75 ours_basec + 0.25 teammate + late dose rule; byproduct 0.4 refined note + 0.4 note + 0.1 ours_basec + 0.1 teammate | substrate **6.35 / 6.45**, byproduct **7.86 / 7.79** | pending | `python src/final_submission.py --base3` |

## Substrate investigation (22 Nov, after base3)
* Current substrate error is already low: mean absolute error 0.12 g/L in CV (team note target: 0.25 g/L; its per-day
  baseline 0.28 g/L). No leakage of VCD errors: corr(substrate residual, VCD residual) = 0.05.
* Feed timing: Feed B always 0.01-0.05 d AFTER the sample (train and test); time since the last dose / interval length
  correlate with S within a day (-0.2 to -0.6) but adding them to the ridge is worse (already carried by t and Vt).
* Initial substrate deviation: S_j - mean_j = a_j (S0 - mean0) with a_1 = 0.73, fading to ~0 by day 6: Feed B acts as a
  feedback controller that pulls S back. Batch54 (S0 = 4.16, -3.8 sd) is predicted consistently with this law.
* Mass balance (Feed B / V, substrate feed / V, S0 V0 / V, IVCD) explains only 25-50 % of the substrate deviations
  between batches even with the TRUE VCD; as ratio features in the ridge: no gain. The rest = controller tracking
  error + batch-specific uptake; the hidden base volume (lactate overflow) is the only extra uptake signal found.
* LB noise: per-batch CV substrate MAPE has sd 3.4 -> a 12-batch mean has sd ~1.0. Given the test composition
  (3 mid-run bolus batches, 5 big late doses) the expected substrate MAPE is ~6.1; LB 8.0 is ~2 sd above, i.e.
  probably 1-2 hard test batches like train Batch46 / 17 / 42 (persistent offsets of +-0.6 g/L from day 4).

## Cells (team note on VCD / TCD), 22 Nov
* tcd built as vcd + dead cells (typical dead/viable ratio per day, or per-day ridge on log dcd), alone or blended
  with the current tcd: 6.43 / 6.44 -> 6.43-6.44 (no change); vcd averaged with tcd-derived vcd: no change.
  Our vcd / tcd predictions are already consistent (tcd = max(tcd, vcd)).
* The note's ~6.5 % VCD measurement noise implies a MAPE floor around 5-5.5 %; current CV vcd 6.50 %.
* Final decision: `submissions/blend_base.csv` (LB 7.464) stays the reference submission.

## Substrate: Feed B controller as a sensor (22 Nov evening)
* Residual mining: the OOF substrate residual correlates with the Feed B dose change z_j = log d_j - log d_{j-1}
  with the same sign on all 11 days (0.02-0.37): the controller raises the dose when the measured S was low.
* Ceiling: with the TRUE S_{k-1}, S_k ~ S_{k-1} + d_k/V + d_{k-1}/V predicts S_k within 1.6-4.7 % on days 2-7
  (S_{k-1} alone: 2.5-7.6 %; per-day models: 2.2-7.1 %). The doses carry information on S, but as a relation between
  consecutive days, which per-day regressions cannot use (z_j as a feature or as a global correction: no gain).
* Gaussian (Kalman-type) update of the predicted trajectory with the controller laws as observations
  (`src/substrate_kalman.py`, `src/experiments/controller_kalman.py`): all 30 settings tried improve the substrate on
  both seed groups; chosen on seeds 0-2: observation days 1, 2, 4-7, correlation shrinkage 0.6, noise x1.5.
  On the blend_base substrate: 6.70 / 6.81 -> 6.62 / 6.73 (gain on all 6 seeds); on the base3 substrate:
  6.35 / 6.45 -> 6.24 / 6.36. Late days (11-13) are not helped (controller laws poor after the substrate boluses).
* `submissions/blend_base_kalman.csv` = blend_base (LB 7.464) with the updated substrate
  (`python src/final_submission.py --kalman`); test substrate changes: median 0.3 %, max 4.5 %.

## While the user was away (22 Nov, after LB 7.438 with blend_base_kalman.csv)
| Try | CV (seeds 0-2 / 3-5) | Verdict |
|---|---|---|
| Controller update: substrate-feed (Feed A) dose added to the laws; observation days re-chosen | days 1,2,4-8,11: 6.578 / 6.721 vs 6.616 / 6.733 | kept |
| shrink / noise / log-space laws (`controller_kalman2.py`) | best shrink 0.4, noise x1.5, linear: **6.574 / 6.719**; log-space better on 3-5 only | kept (linear) |
| prior weights ours / teammate under the update | 0.5 / 0.5 and 0.75 / 0.25 equal | 0.5 kept |
| laws with slopes pooled over days 4-8 (more data per law) | worse or equal | dropped |
| sensors for byproduct (hidden volume per interval, Feed C) and VCD (Feed B doses): ceiling with the TRUE previous value | no gain beyond the previous value | dropped |
| joint substrate + byproduct update (cross-correlated errors) | cross-correlation only 0.17 on average | dropped |
| cells: FPCA smoothing of the final blend; re-weighting with time-weighted SVR / kernel ridge | worse / +0.002 | dropped |
Files: `submissions/blend_base_kalman2.csv` (`python src/final_submission.py --kalman2`) and
`submissions/blend_base_kalman2_byp3.csv` (same + base3 refined byproduct, CV byproduct -0.04; LB of base3 unknown).

## Control-rule constraints inside the controller update (22 Nov, after LB 7.410 with blend_base_kalman2.csv)
* LB: `blend_base_kalman2.csv` = **7.410** (substrate 7.898 -> 7.788): controller-update gains transferred to the LB
  more than in CV, twice.
* Late dose classes as pseudo-observations (violated bound -> observation S_k = bound, sd 0.03 g/L, iterated) so the
  correction propagates to neighbouring days: substrate 6.574 / 6.719 -> **6.494 / 6.612** (sd 0.01: same).
  File: `submissions/blend_base_kalman3_late.csv` (`python src/final_submission.py --kalman3`).
* + mid-run bolus constraints (S(j0) = 1.85, S(j0+1) = 3.1, window floor 1.91; sd 0.1): -> **6.468 / 6.554**
  (sd 0.2: 6.480 / 6.578; floor only: 6.484 / 6.599). File: `submissions/blend_base_kalman3_full.csv`
  (`--kalman3 --mid`). Caution: the same mid-run rule applied as a post-processing was slightly worse on the LB.
* `src/substrate_kalman.py` reproduces `src/experiments/controller_kalman3.py` exactly (max diff 0.0).

## After LB 7.300 (blend_base_kalman3_full.csv: substrate 7.349)
* Re-tuned constrained update (`controller_kalman4.py`, chosen on seeds 0-2): days 1,2,4-8,11,12, shrink 0.4,
  noise x1.25, tau_mid 0.05, late-rule quantile 0.1: substrate 6.468 / 6.554 -> **6.410 / 6.487**.
* Byproduct residual mining: control-regime indicators with a constant sign on all days (big11 +, trig -).
  In the hidden-base Huber model: + trig (mid-run bolus batch) 7.903 / 7.833 -> 7.810 / 7.782; + Huber epsilon 2.0
  -> **7.752 / 7.725**; + big11 / big12: worse; nested pooled shifts: no gain; trig_early (4 batches only): not used.
* `submissions/blend_kalman4.csv` (`python src/final_submission.py --kalman4`): both changes; vcd / tcd unchanged.
  `submissions/blend_kalman4_substrate_only.csv`: only the substrate change (diagnostic).

## Team proposal: median_d + lambda * (ODE sim deviation) + ridge(x_b) on residuals (`physics_deviation.py`)
* sim = teammate ODE fitted fold by fold (their_oof *_mech, honest); lambda in {0, .25, .5, .75, 1}; 9 batch features.
* Standalone (seeds 0-2): vcd 7.07-7.29, tcd 7.03-7.23, substrate 9.44-10.77, byproduct 10.42-11.29 vs current
  6.51 / 6.43 / 6.47 / 7.90. As a blend member (w = 0.1): worse for every target and every lambda. Not used.
* blend_kalman4.csv and blend_kalman4_substrate_only.csv were worse than 7.300 on the LB: kalman3_full stays.

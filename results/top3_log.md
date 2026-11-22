# Top-3 log (Team 10)

## Leaderboard, 22 Nov 2026 ~11:45 UTC (get_leaderboard, read-only)
| rank | team | vcd | tcd | substrate | byproduct | mean MAPE |
|---|---|---|---|---|---|---|
| 1 | Team 7 | 6.627 | 6.569 | 6.326 | 7.314 | 6.709 |
| 2 | Team 04 | 6.747 | 6.550 | 6.633 | 6.969 | 6.725 |
| 3 | Team 15 | 6.609 | 6.463 | 6.479 | 7.457 | 6.752 |
| 4 | Team 23 | 6.534 | 6.439 | 6.737 | 8.258 | 6.992 |
| 7 | **Team 10** | 7.164 | 7.044 | 7.349 | 7.644 | 7.300 |
Metric: MAPE (%) per target, plain mean of the 4. Gap to 3rd place: **0.548** (all four targets; cells +0.5-0.6).
Our best file: `submissions/blend_base_kalman3_full.csv` (114 attempts so far).

## CV vs leaderboard (7 folds by batch x 6 seeds, same metric, same 13 time points)
| file | CV vcd / tcd / S / P | LB vcd / tcd / S / P |
|---|---|---|
| blend_base_kalman3_full | 6.50 / 6.43 / 6.47-6.55 / 7.83-7.90 | 7.164 / 7.044 / 7.349 / 7.644 |
Our LB is ~0.6 worse than CV on the cells and ~0.8 on the substrate, but better on the byproduct.
The other teams' LB cell scores (6.4-6.8) are close to our CV: our cell models generalise worse to the test set.

## Submission history (leaderboard, mean MAPE; vcd / tcd / substrate / byproduct)
| file | LB | per target | what changed |
|---|---|---|---|
| ridge_blend | 8.119 | 7.665 / 7.616 / 8.417 / 8.777 | per-time ridge on log targets |
| hybrid_ensemble | 7.938 | 7.279 / 7.245 / 8.454 / 8.772 | + baselines, FPCA, hybrid ODE, substrate rule |
| blend_team | 7.725 | 7.183 / 7.178 / 8.297 / 8.241 | geometric blend with the teammate model |
| blend_nonlinear | 7.686 | 7.164 / 7.044 / 8.297 / 8.241 | pooled-time SVR + trees for the cells |
| blend_nonlinear_dose | 7.613 | 7.164 / 7.044 / 8.003 / 8.241 | late substrate dose rule |
| blend_base | 7.464 | 7.164 / 7.044 / 8.003 / 7.644 | hidden base volume -> byproduct |
| blend_base_kalman | 7.438 | ... / 7.898 / 7.644 | Feed B controller as substrate sensor (Kalman update) |
| blend_base_kalman2 | 7.410 | ... / 7.788 / ... | tuned update (days 8, 11, Feed A in laws) |
| blend_base_kalman3_late | 7.389 | ... / 7.706 / ... | late-dose constraints inside the update |
| **blend_base_kalman3_full** | **7.300** | 7.164 / 7.044 / 7.349 / 7.644 | + mid-run bolus constraints inside the update |
| blend_kalman4 (+ substrate-only variant) | worse | - | fine re-tuning + trig in byproduct: did not transfer |
(base2 / base3 / nonlinear_rules: worse than the reference at the time.)

## This session (after reading the leaderboard: rank 7, gap 0.548)
| idea | evidence | verdict |
|---|---|---|
| initial lactate (log_byproduct0) makes the cell models extrapolate (4 test batches at z 2.2-2.7, train max 2.23) | removing it: extreme-lactate batches 7.9 -> 9.2 % (ridge), 7.3 -> 9.4 % (SVR) | it HELPS; kept |
| clip inputs to the training range | ridge 0, SVR -0.02 | negligible |
| high-lactate train batches are under-predicted (-4 % for z > 1.5): de-shrink / slope correction | kappa 1.1: -0.01; slope: 0 | no gain |
| linear models for out-of-support batches | on the 6 most extreme train batches the blend (7.18) beats linear (7.45) and SVR (7.39) | no |
| sampling-time offsets test vs train | identical (< 0.02 d); residual unrelated | no |
| Feed B doses / hidden volume / Feed C as a VCD sensor | batch offset correlations only at the multiple-testing noise level | no |
| batch-number drift (campaign effects) | corr with batch number ~0; ID neighbours uncorrelated | no |
| team proposal median + lambda*(ODE sim) + ridge | standalone 7.1 / 7.0 / 9.4 / 10.4; worse in every blend | no |
Expected test VCD given the test composition (4-5 out-of-support batches): ~6.7 vs LB 7.16; the remaining
cell gap is unexplained by everything checked above.
Git: a merge started by someone else is in progress (MERGE_HEAD cee934a3, 87 staged files incl. copies of the
challenge data in new/ and the deletion of src/v2.py): nothing committed, waiting for that merge to be finished.

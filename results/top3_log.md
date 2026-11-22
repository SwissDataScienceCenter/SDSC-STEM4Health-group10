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

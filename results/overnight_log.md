# Overnight log — Decode the Bioreactor

- **Start:** Sat Nov 21 23:16:49 UTC 2026
- **Deadline (start + 55 min):** Sun Nov 22 00:11:49 UTC 2026
- **No new experiment after:** 23:56:49 UTC (15 min before deadline)
- Baseline to beat: `submissions/ridge_blend.csv`, ~7.52 % mean MAPE (LOBO CV)

## 0. LOBO harness + reference (23:18)
- Note: the 7.52 % of ridge_blend was a 7-fold grouped CV (seeds 0-2), not LOBO. New shared harness
  `src/experiments/common.py`: leave-one-batch-out over the 49 train batches, MAPE per target pooled over points,
  same splits for every experiment. OOF predictions saved in `results/overnight/<name>_oof.csv`.
- Reference reproduces `submissions/ridge_blend.csv` exactly (max rel. diff 0).

| model | vcd | tcd | substrate | byproduct | mean |
|---|---|---|---|---|---|
| ref ridge_blend (current submission) | 6.892 | 6.858 | 7.042 | 9.104 | **7.474** |
| ref + FPCA vcd/tcd (2 modes) | 6.805 | 6.750 | 7.042 | 9.104 | 7.425 |

## 3. Better features (subagent, src/experiments/features.py) — DEAD END
Each group added to every target's reference features, same ridge + 0.2 blend.

| variant | vcd | tcd | substrate | byproduct | mean |
|---|---|---|---|---|---|
| G1 feed/V fractions, dilution V0/Vt, log V/V0, added-substrate conc. | 6.938 | 6.876 | 7.115 | 9.187 | 7.529 |
| G2 time since last S feed, Feed B last 2 d, B 1st/2nd half, n S boluses | 6.920 | 6.879 | 7.445 | 9.399 | 7.661 |
| G3 log t, log1p cumulative B / S | 6.936 | 6.903 | 7.079 | 9.182 | 7.525 |
| G4 IC ratios: log viability0, log S0/P0, log dcd0 | 6.872 | 6.835 | 7.623 | 9.333 | 7.666 |

Conclusion: no group improves any target by > 0.05; the reference feature set is already near-optimal for ridge.
(Feed C/D features deliberately excluded: they failed on the leaderboard before.)

## 4a. Smoothing across time: FPCA projection of the predicted log trajectories (src/experiments/smooth_pls.py)
Same K for the 4 targets per run, then 0.2 blend. K per target then chosen by nested selection (per held-out
batch, the K best on the other 48 batches' OOF predictions).

| variant | vcd | tcd | substrate | byproduct | mean |
|---|---|---|---|---|---|
| K=1 | 6.749 | 6.686 | 10.229 | 9.267 | 8.233 |
| K=2 | 6.805 | 6.750 | 9.058 | 9.174 | 7.947 |
| K=3 | 6.804 | 6.749 | 8.888 | 9.171 | 7.903 |
| K=4 | 6.818 | 6.775 | 8.128 | 9.121 | 7.710 |
| nested choice among {none, 1..4} | 6.749 | 6.686 | 7.042 | 9.104 | **7.395** |

Nested picks: vcd and tcd -> K=1 in 49/49 folds; substrate and byproduct -> no projection in 49/49 folds (stable).
Conclusion: projecting vcd/tcd on the mean curve + 1 mode helps (-0.14 / -0.17); substrate must not be projected
(feed jumps). Mean gain -0.079 alone (< 0.1 threshold) -> kept as a candidate for the combination.

## PLS (cheap ensemble member)
| PLS n_comp | vcd | tcd | substrate | byproduct | mean |
|---|---|---|---|---|---|
| 2 | 6.986 | 6.926 | 8.774 | 10.717 | 8.351 |
| 3 | 7.086 | 7.042 | 8.460 | 10.774 | 8.340 |
| 5 | 7.105 | 7.035 | 7.872 | 9.877 | 7.972 |
Conclusion: PLS clearly worse than ridge on every target; included only as a candidate member in the nested blend.

## 2a. Nested-CV ensembles of cheap members (src/experiments/members.py + ensemble.py) (23:24)
Members (unblended LOBO OOF): ridge_s = reference ridge with vcd/tcd projected on 1 FPCA mode; ridge_s_k025 = same,
penalty x0.25; base_scaled = VCD0-scaled mean curve; nearest = 8 nearest training batches (geometric mean).
Weights: per target, simplex grid step 0.1, chosen per held-out batch on the other 48 (nested).

| ensemble (nested weights) | vcd | tcd | substrate | byproduct | mean |
|---|---|---|---|---|---|
| members alone: ridge_s / ridge_s_k025 / base / nearest (vcd) | 6.846 / 6.774 / 8.155 / 6.827 | | | | |
| ref + fpca_K1 + pls5 | 6.756 | 6.687 | 7.042 | 9.092 | 7.394 |
| ridge_s + base | 6.749 | 6.686 | 7.045 | 9.145 | 7.406 |
| ridge_s + ridge_s_k025 + base | 6.819 | 6.762 | 7.101 | 9.151 | 7.458 |
| ridge_s + ridge_s_k025 + base + nearest | **6.636** | **6.526** | 7.101 | 9.148 | 7.353 |

Weights fit on all 49 (4-member): vcd 0.5 k025 / 0.5 nearest; tcd 0.5 k025 / 0.1 base / 0.4 nearest;
substrate 0.9 k025 / 0.1 base; byproduct 0.8 k025 / 0.2 nearest.
Conclusion: nearest-batches is a strong complement for cell densities (-0.11 / -0.16 vs FPCA alone).
For substrate/byproduct, tuning weights in nested CV is WORSE than the reference fixed 0.2 blend (selection
noise) -> keep the reference there. PLS brings nothing (dropped).

## 4b. Direct curve-parameter model (subagent, src/experiments/curve_params.py) — DEAD END
Per target: PCA of the log trajectories (refit per fold), one multi-output ridge from batch-level features
(static + Vt / cumulative feeds at j = 3, 6, 9, 12, 13 + sampling times) to the K scores, then 0.2 blend.
curve_resid = 50/50 log average with the per-time ridge.

| variant | vcd | tcd | substrate | byproduct | mean |
|---|---|---|---|---|---|
| curve_pca_K2 | 6.829 | 6.749 | 9.235 | 9.899 | 8.178 |
| curve_pca_K3 | 6.847 | 6.747 | 9.119 | 9.931 | 8.161 |
| curve_pca_K4 | 6.892 | 6.788 | 8.435 | 9.974 | 8.022 |
| curve_resid_K2 | 6.798 | 6.747 | 7.739 | 9.325 | 7.652 |
| curve_resid_K3 | 6.809 | 6.744 | 7.687 | 9.356 | 7.649 |
| curve_resid_K4 | 6.835 | 6.775 | 7.398 | 9.387 | 7.599 |

Conclusion: on vcd/tcd no better than projecting the per-time ridge (4a); much worse on substrate/byproduct
(trajectories too smooth for the feed jumps). Not kept.

## 2b. Fixed (untuned) weights for vcd/tcd, for robustness (23:25)
| vcd/tcd blend (geometric, fixed) | vcd | tcd |
|---|---|---|
| 0.8 ridge_s + 0.2 base | 6.749 | 6.686 |
| 0.5 ridge_s_k025 + 0.5 nearest | 6.564 | 6.493 |
| 0.45 k025 + 0.45 nearest + 0.1 base | 6.570 | 6.487 |
| 0.7 k025 + 0.15 nearest + 0.15 base (team's earlier BLEND) | 6.638 | 6.564 |
| 0.5 ridge_s + 0.5 nearest | 6.603 | 6.531 |
These 5 settings are compared on LOBO itself, so slightly optimistic: the honest number is the nested 6.636 / 6.526.

## First improvement committed (23:26): `submissions/overnight_v1.csv`, commit 54d020c
vcd/tcd = nested 4-member ensemble (weights refit on 49: vcd 0.5 ridge_s_k025 + 0.5 nearest; tcd 0.5 / 0.1 base / 0.4),
substrate/byproduct = reference. Honest LOBO: vcd 6.636, tcd 6.526, substrate 7.042, byproduct 9.104, **mean 7.327**
(-0.147 vs 7.474). Validated: 156 rows, no NaN, all > 0. Not submitted.

## 2c. Nearest-batches k (src/experiments/nearest_k.py) (23:27)
| member | vcd | tcd |
|---|---|---|
| nearest k=4 / 8 / 12 / 16 | 7.319 / 6.827 / 6.847 / 6.906 | 7.237 / 6.733 / 6.742 / 6.817 |

| nested ensemble | vcd | tcd | substrate | byproduct | mean |
|---|---|---|---|---|---|
| ridge_s_k025 + nearest4 + nearest8 | 6.564 | 6.493 | 6.985 | 9.141 | 7.296 |
| ridge_s_k025 + nearest8 + nearest16 | 6.572 | 6.533 | 6.984 | 9.061 | 7.287 |
Weights refit on 49 for vcd and tcd: 0.5 ridge_s_k025 + 0.5 nearest8 in both (k=4/16 get 0).
Conclusion: k=8 confirmed; for cell densities the simplest robust choice is the plain 50/50 geometric mean of
ridge_s_k025 and nearest8 (the baseline member gets ~0). Substrate/byproduct gains < 0.1 -> keep reference.

## 5. Hybrid serial model (subagent, src/experiments/serial_hybrid.py) — DEAD END
Specific rates mu, kd, qS, qP computed from mass balances on total amounts (Feed B 50, substrate feed 250 assumed),
ridge from state + feeds in interval + cumulative feeds (teacher forcing), rolled forward from t = 0 through the
batch's own sampling times with its known feeds and volume.

| variant | vcd | tcd | substrate | byproduct | mean |
|---|---|---|---|---|---|
| serial_raw | 8.387 | 8.291 | 15.708 | 21.365 | 13.437 |
| serial_blend (0.2 with baseline) | 7.715 | 7.609 | 13.975 | 18.190 | 11.872 |
| serial_noisy (1 DAgger round) | 8.195 | 8.095 | 15.742 | 33.938 | 16.492 |

Conclusion: errors compound over 13 steps (esp. substrate/byproduct, which integrate small rate biases);
DAgger destabilises qP. Rate-model insight: mu decreases with log Xv (density-dependent growth) and with
cumulative Feed B; kd essentially unpredictable from state. Not kept.

## 1. Gaussian process, same conditions as final model (subagent, src/experiments/gp.py) — DEAD END
Per (target, j) on log target, per-target features, scaler inside the fold, 0.2 blend; normalize_y, 0 restarts.

| variant | vcd | tcd | substrate | byproduct | mean |
|---|---|---|---|---|---|
| gp_resid: ridge + GP(Const*RBF-ARD + White) on ridge log-residuals | 7.528 | 7.369 | 7.142 | 9.312 | 7.837 |
| gp_resid unblended | 7.842 | 7.660 | 7.311 | 9.430 | 8.061 |
| gp_pure: Const*DotProduct + Const*RBF-ARD + White | 7.464 | 7.426 | 8.074 | 10.771 | 8.434 |
| gp_pure unblended | 7.912 | 7.882 | 8.250 | 11.058 | 8.775 |

Conclusion: no GP variant wins any target (the earlier "better on byproduct" impression is not confirmed under
LOBO with the final model's features and blend: 9.31 vs 9.10). ARD kernels overfit with 48 batches per fit.
Nested ensembles with GP members: ref + gp_resid -> 7.482; ridge_s_k025 + nearest + gp_pure + gp_resid -> 7.353
(GP gets weight 0 on vcd/tcd, and the nested substrate/byproduct scores are worse than the reference). Not kept.

## 2d. Team hybrid ODE as a member (src/experiments/ode_member.py) (23:32)
ode_paramnet v3 (6 batch-specific parameters predicted by ridge), per-batch fit + param-net refit in each LOBO fold
(shared v3 parameters fitted once on all 49 batches: mild leakage). Alone: vcd 7.006, tcd 6.877, substrate 19.2,
byproduct 19.5. Nested ensemble ridge_s_k025 + nearest + ode_v3: vcd 6.540, tcd 6.499 (weights 0.3/0.4/0.3),
substrate 7.003, byproduct 9.061, mean 7.276.
Conclusion: +ODE gains 0.00-0.10 on vcd/tcd only, < 0.1 threshold, + leakage caveat + 28 s/fold -> not kept.

## 2e. Penalty x0.25 for substrate/byproduct (team's earlier ALPHA_SCALE, fixed 0.2 blend, not tuned tonight)
substrate 6.980 (-0.062), byproduct 9.051 (-0.053): each < 0.1 -> reference kept (option for later).

## Robustness of the final choice (overnight_v2: vcd/tcd = 0.5 ridge_s_k025 + 0.5 nearest8)
- per batch: better than the reference on 34/49 (vcd) and 35/49 (tcd) batches; better or equal at every j = 1..13.
- test batches vs training: mean distance to their 8 nearest training batches 2.51 (train LOBO 2.21, max 5.13 vs
  test max 4.27) -> the nearest-batches member is not extrapolating for the test set.

## FINAL (23:34): submissions/overnight_v2.csv — see results/overnight_report.md

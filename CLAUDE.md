# SDSC STEM4Health hackathon: team 10, "Decode the Bioreactor" (Project 1 - DTB, J&J Innovative Medicine)

Goal: model fed-batch cell culture dynamics (growth, substrate consumption, byproduct formation, feeding/dilution)
from historical batches and **predict unseen runs from their initial conditions + feed profile**.
Interpretability / process insight matters as much as accuracy. Organisers encourage growth kinetics (Monod, ...)
and mass-balance formulations.

## Folder layout (inside ~/work)
- `SDSC-STEM4Health-group10/`: team git repo. ALL code, notebooks and figures go here
  (anything outside it is lost when the session shuts down).
  - `src/data.py`: reusable loading helpers (raw log, measurements, feeds, volume, train/test ids, template).
  - `notebooks/01_eda.ipynb`: exploratory analysis. `figures/eda/`: its saved figures.
- `input-data/Project 1 - DTB/`: challenge data. **READ-ONLY. Never copy, modify or commit it.**
  Read it in place (`data.DATA_DIR`, overridable with env var `DTB_DATA_DIR`).
- `team-workspace/`: large files (models, caches), outside git.
- Python: `/home/renku/work/.venv/bin/python` or the conda env (`jupyter` available).
  pandas, numpy, scipy, matplotlib, plotly, scikit-learn are installed.

## Data
Files in `input-data/Project 1 - DTB/`:
- `bioreaction_remaining.csv`: 2858 rows, 61 batches (`Batch1`…`Batch61`), sparse **event log**.
- `bioreaction_template.csv`: submission format, 156 rows: `batch_id, elapsed_time, vcd, tcd, substrate, byproduct`.
- `Leaderboard_template.ipynb`: `get_leaderboard()` and `submit(df, user="Team XX")` (POST to organisers' API;
  metrics computed server-side).

Columns of the event log:
| column | meaning | notes |
|---|---|---|
| `batch_id` | batch | |
| `elapsed_time` | time since inoculation | days, 0 → ~12.9 |
| `vcd`, `tcd` | viable / total cell density | likely 10⁶ cells/mL; **DCD = tcd − vcd** (always ≥ 0) |
| `substrate` | substrate concentration | likely g/L (glucose-like), 0.3–5 |
| `byproduct` | byproduct concentration | likely g/L (lactate-like), 0.2–3 |
| `feed_substrate`, `feed_b`, `feed_c`, `feed_d` | **bolus volume added at that instant** | same unit as `volume`; not cumulative, not a rate |
| `volume` | reactor volume | given at every sampling time (also on to_predict rows); 10.45 → ~16.8 |
| `to_predict` | row whose targets are hidden | test batches only |

Units are **not documented**; the units above are assumptions.

Row types (exclusive): measurement (all 4 targets + volume), feed (exactly one feed column), to_predict
(time + volume only). Missing values are structural only.

Sampling: 14 samples per batch, ~daily (interval 0.85–1.10 d), timestamps differ across batches → use continuous time.

Feed encoding (verified): between samples, ΔV = Σ feed volumes − ~0.04 (sampling/evaporation loss), corr 0.997.
Bolus of volume ΔV_f at concentration C_f: `V⁺ = V + ΔV_f`, `C⁺ = (C·V + C_f·ΔV_f)/V⁺` (cells: C_f = 0).
Feed concentrations are NOT given and must be estimated.
- Feed B: 12 daily boluses (day ~1 → 12), total ≈ 6.6 (≈ 60 % of V₀), size depends on the last measured VCD;
  contains substrate (≈ 50 units/volume unit by mass balance).
- Substrate feed: concentrated (≈ 250 units/volume unit), 2–3 boluses late (day ~11–12.5), when substrate < ~2.
- Feed C: tiny fixed boluses (0.00085 / 0.0018), 11–35 per batch, on-demand additive; negligible volume.
- Feed D: 3 boluses at days 5, 8, 10 (~0.011–0.017); negligible volume, composition unknown.

Split imposed by organisers: 12 **test** batches (Batch5, 6, 12, 24, 32, 35, 40, 48, 54, 56, 57, 60) have only the
t = 0 measurement + full feed profile + volume; predict their 13 later samples (= template rows).
49 **train** batches are fully measured.

Leaderboard metric (told by the organisers, Nov 21): **MAPE** = mean |pred − true| / |true| (%).
Relative errors weigh low values heavily (days 1–3 VCD/TCD, low byproduct/substrate), so model selection
and fitting should target MAPE on the 4 submitted variables (vcd, tcd, substrate, byproduct).
How the 4 variables are combined server-side is not documented (assumed: plain average).

## Rules
- Honest evaluation: split **by batch** (never points of one batch in both train and validation);
  leave-one-batch-out or grouped K-fold. A validation/test run is predicted **only from its initial conditions
  and feed profile (+ given volume)**, never from its intermediate measurements.
- Metrics: RMSE and relative error per variable (vcd, tcd/dcd, substrate, byproduct), plus
  predicted-vs-measured plots per test batch.
- Clear, commented Python code using pandas/numpy/scipy/matplotlib or plotly. If a package is missing,
  **ask before installing**.
- **Ask before any git commit / push.**
- Never write to `input-data/`; never commit data. Large artefacts go in `team-workspace/`.

"""Final team submission: geometric blend of our hybrid regression model with the teammate's model.

Usage (from the repo root):
    python src/run_regression.py          # -> submissions/hybrid_ensemble.csv  (our model, LB 7.938)
    python src/v2.py                      # -> teammate's model (LB 7.743), same as src/submission.csv
    python src/blend_team.py              # -> submissions/blend_team.csv       (LB 7.725)

For every target, prediction = exp((1 - w) * log(ours) + w * log(teammate)): a weighted geometric mean,
consistent with the MAPE metric (relative errors). Weights w on the teammate's model were chosen by grouped
7-fold cross-validation by batch (seeds 0-2) and checked on fresh seeds (3-5). Byproduct uses the teammate's
model only (none of our byproduct variants beat it in CV). Total cell density is kept >= viable cell density.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D

REPO = Path(__file__).resolve().parents[1]
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
OURS = REPO / "submissions" / "hybrid_ensemble.csv"
TEAMMATE = REPO / "src" / "submission.csv"          # output of src/v2.py
OUT = REPO / "submissions" / "blend_team.csv"
W_TEAMMATE = {"vcd": 0.6, "tcd": 0.5, "substrate": 0.5, "byproduct": 1.0}


def blend(ours: pd.DataFrame, teammate: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = teammate[["batch_id", "elapsed_time"]].copy()
    for v in TARGETS:
        w = weights[v]
        out[v] = np.exp((1 - w) * np.log(ours[v]) + w * np.log(teammate[v]))
    out["tcd"] = np.maximum(out["tcd"], out["vcd"])
    return out


def main():
    tpl = D.load_template()
    ours, teammate = pd.read_csv(OURS), pd.read_csv(TEAMMATE)
    key = ["batch_id", "elapsed_time"]
    for name, df in [("ours", ours), ("teammate", teammate)]:
        assert len(df) == len(tpl), f"{name}: {len(df)} rows, template has {len(tpl)}"
        assert (df["batch_id"].to_numpy() == tpl["batch_id"].to_numpy()).all(), f"{name}: batch order differs"
        assert np.allclose(df["elapsed_time"], tpl["elapsed_time"]), f"{name}: time grid differs"
        assert (df[TARGETS] > 0).all().all(), f"{name}: non-positive predictions"
    sub = blend(ours, teammate, W_TEAMMATE)
    assert sub[TARGETS].notna().all().all() and (sub[TARGETS] > 0).all().all()
    sub.to_csv(OUT, index=False)
    print(f"submission written: {OUT.relative_to(REPO)} ({len(sub)} rows), weights on teammate model: {W_TEAMMATE}")


if __name__ == "__main__":
    main()

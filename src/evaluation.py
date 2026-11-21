"""Batch-wise evaluation: grouped folds, metrics and predicted-vs-measured plots.

Rule: a validation batch is predicted ONLY from its t=0 state and its feed profile (+ given volume);
its later measurements are used solely to compute the error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EVAL_VARS = ["vcd", "tcd", "dcd", "substrate", "byproduct"]


def make_folds(batch_ids: list[str], n_folds: int = 7, seed: int = 0) -> list[list[str]]:
    """Random partition of batches into n_folds groups (no batch in two folds)."""
    rng = np.random.default_rng(seed)
    ids = np.array(batch_ids)[rng.permutation(len(batch_ids))]
    return [sorted(f.tolist(), key=lambda s: int(s[5:])) for f in np.array_split(ids, n_folds)]


def cross_validate(raw, folds, fit_predict) -> pd.DataFrame:
    """Out-of-fold predictions. `fit_predict(raw, train_ids, val_ids)` returns a DataFrame with
    batch_id, elapsed_time and the predicted variables for the validation batches."""
    all_ids = [b for f in folds for b in f]
    out = []
    for i, val in enumerate(folds):
        train = [b for b in all_ids if b not in val]
        p = fit_predict(raw, train, val)
        p["fold"] = i
        out.append(p)
    return pd.concat(out, ignore_index=True)


def attach_truth(pred: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    """Join predictions (columns renamed *_pred) with measured values on (batch_id, elapsed_time)."""
    key = lambda d: d.assign(_t=d["elapsed_time"].round(6))
    p = key(pred).rename(columns={v: f"{v}_pred" for v in EVAL_VARS})
    m = key(meas)[["batch_id", "_t", *EVAL_VARS]]
    return p.drop(columns="elapsed_time").merge(m, on=["batch_id", "_t"]).rename(columns={"_t": "elapsed_time"})


def metrics(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """RMSE, relative RMSE (RMSE / mean measured, %) and mean absolute relative error (%) per variable."""
    def _m(g):
        rows = {}
        for v in EVAL_VARS:
            e = g[f"{v}_pred"] - g[v]
            rows[v] = {
                "RMSE": np.sqrt(np.mean(e ** 2)),
                "relRMSE_%": 100 * np.sqrt(np.mean(e ** 2)) / g[v].mean(),
                "MARE_%": 100 * np.mean(np.abs(e) / np.maximum(np.abs(g[v]), 1e-9)),
            }
        return pd.DataFrame(rows).T
    if by is None:
        return _m(df)
    return pd.concat({k: _m(g) for k, g in df.groupby(by)}, names=[by, "variable"])


TARGETS = ["vcd", "tcd", "substrate", "byproduct"]   # variables scored by the leaderboard


def mape_summary(df: pd.DataFrame) -> pd.Series:
    """Leaderboard metric: MAPE (%) per submitted variable + plain average over the 4 variables."""
    out = {v: 100 * np.mean(np.abs(df[f"{v}_pred"] - df[v]) / np.abs(df[v])) for v in TARGETS}
    out["mean_4_targets"] = float(np.mean(list(out.values())))
    return pd.Series(out, name="MAPE_%")


def plot_pred_vs_meas(df: pd.DataFrame, batches: list[str], variables=("vcd", "dcd", "substrate", "byproduct"),
                      models: dict | None = None, ncols: int = 4):
    """Per-batch time courses: measured points vs predicted lines (one or several models).

    `models` maps label -> DataFrame from attach_truth (default: {"model": df})."""
    models = models or {"model": df}
    fig, axes = plt.subplots(len(batches), len(variables), figsize=(3.6 * len(variables), 2.4 * len(batches)),
                             sharex=True, squeeze=False)
    for i, b in enumerate(batches):
        for j, v in enumerate(variables):
            ax = axes[i, j]
            ref = next(iter(models.values()))
            g = ref[ref.batch_id == b].sort_values("elapsed_time")
            ax.plot(g.elapsed_time, g[v], "ko", ms=3, label="measured")
            for lab, d in models.items():
                g = d[d.batch_id == b].sort_values("elapsed_time")
                ax.plot(g.elapsed_time, g[f"{v}_pred"], "-", lw=1.3, label=lab)
            ax.set_title(f"{b} - {v}", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time [d]")
    fig.tight_layout()
    return fig


def parity_plot(df: pd.DataFrame, variables=("vcd", "dcd", "substrate", "byproduct"), title=""):
    """Predicted vs measured scatter, all validation points, coloured by time."""
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 3.6))
    for ax, v in zip(axes, variables):
        sc = ax.scatter(df[v], df[f"{v}_pred"], c=df["elapsed_time"], s=6, cmap="viridis")
        lo, hi = min(df[v].min(), df[f"{v}_pred"].min()), max(df[v].max(), df[f"{v}_pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"measured {v}"); ax.set_ylabel(f"predicted {v}")
    fig.colorbar(sc, ax=axes, label="time [d]")
    fig.suptitle(title)
    return fig

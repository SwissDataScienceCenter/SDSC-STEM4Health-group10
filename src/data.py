"""Data loading utilities for the "Decode the Bioreactor" (Project 1 - DTB) challenge.

The raw file `bioreaction_remaining.csv` is a *sparse event log*: every row is
either
  * a MEASUREMENT row (vcd, tcd, substrate, byproduct and reactor volume), or
  * a FEED row (exactly one of the four feed columns is filled), or
  * a TO-PREDICT row (test batches only: time + volume, targets hidden).

Feeds are BOLUS VOLUMES added at that instant, in the same unit as `volume`
(checked: between two samples, delta V ~= sum of feed volumes - ~0.04 removed
per sample, correlation 0.997). They are NOT cumulative and NOT flow rates.

The input data are READ-ONLY: this module never writes anything there.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths and column names
# --------------------------------------------------------------------------
# Repo layout: ~/work/SDSC-STEM4Health-group10/src/data.py
#              ~/work/input-data/Project 1 - DTB/
# Can be overridden with the DTB_DATA_DIR environment variable.
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    os.environ.get("DTB_DATA_DIR", REPO_DIR.parent / "input-data" / "Project 1 - DTB")
)
RAW_FILE = "bioreaction_remaining.csv"
TEMPLATE_FILE = "bioreaction_template.csv"

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]  # columns to predict
STATES = ["vcd", "dcd", "tcd", "substrate", "byproduct"]  # incl. derived dcd
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]

FEED_LABELS = {
    "feed_substrate": "Substrate feed",
    "feed_b": "Feed B",
    "feed_c": "Feed C",
    "feed_d": "Feed D",
}


# --------------------------------------------------------------------------
# Raw loading
# --------------------------------------------------------------------------
def load_raw(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load the raw event log, sorted by batch and time.

    Adds:
      * `batch_num` (int) for natural sorting (Batch2 before Batch10),
      * `row_type` in {"measurement", "feed", "to_predict"}.
    """
    df = pd.read_csv(Path(data_dir) / RAW_FILE, dtype={"batch_id": str})
    df["to_predict"] = df["to_predict"].astype(bool)
    df["batch_num"] = df["batch_id"].str.extract(r"(\d+)").astype(int)

    is_meas = df[TARGETS].notna().any(axis=1)
    is_feed = df[FEEDS].notna().any(axis=1)
    df["row_type"] = np.select(
        [df["to_predict"], is_meas, is_feed],
        ["to_predict", "measurement", "feed"],
        default="unknown",
    )
    return df.sort_values(["batch_num", "elapsed_time"]).reset_index(drop=True)


def load_template(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Submission template: batch_id, elapsed_time + the 4 targets (zeros)."""
    return pd.read_csv(Path(data_dir) / TEMPLATE_FILE, dtype={"batch_id": str})


# --------------------------------------------------------------------------
# Train / test batches
# --------------------------------------------------------------------------
def _natural(ids) -> list[str]:
    return sorted(set(ids), key=lambda s: int("".join(c for c in s if c.isdigit())))


def test_batches(df: pd.DataFrame) -> list[str]:
    """Batches with hidden targets (only t=0 measured): the leaderboard set."""
    return _natural(df.loc[df["to_predict"], "batch_id"])


def train_batches(df: pd.DataFrame) -> list[str]:
    """Fully measured batches, usable for fitting and leave-batch-out CV."""
    return _natural(set(df["batch_id"]) - set(test_batches(df)))


# --------------------------------------------------------------------------
# Tidy tables
# --------------------------------------------------------------------------
def measurements(df: pd.DataFrame, include_to_predict: bool = False) -> pd.DataFrame:
    """One row per sampling time with states + volume.

    `dcd` (dead cell density) is derived as tcd - vcd and `viability` as vcd/tcd.
    With include_to_predict=True the hidden test rows are kept (targets = NaN),
    which gives the prediction time grid and the measured volume there.
    """
    types = ["measurement", "to_predict"] if include_to_predict else ["measurement"]
    m = df.loc[df["row_type"].isin(types),
               ["batch_id", "batch_num", "elapsed_time", *TARGETS, "volume", "to_predict"]].copy()
    m["dcd"] = m["tcd"] - m["vcd"]
    m["viability"] = m["vcd"] / m["tcd"]
    return m.reset_index(drop=True)


def feeds_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per feed event: batch_id, elapsed_time, feed (name), amount (volume added)."""
    f = df.loc[df["row_type"] == "feed", ["batch_id", "batch_num", "elapsed_time", *FEEDS]]
    f = f.melt(id_vars=["batch_id", "batch_num", "elapsed_time"], value_vars=FEEDS,
               var_name="feed", value_name="amount").dropna(subset=["amount"])
    return f.sort_values(["batch_num", "elapsed_time"]).reset_index(drop=True)


def cumulative_feeds(df: pd.DataFrame, batch_id: str) -> pd.DataFrame:
    """Cumulative added volume of each feed over time for one batch (step functions)."""
    f = df.loc[(df["batch_id"] == batch_id) & (df["row_type"] == "feed"), ["elapsed_time", *FEEDS]]
    return f.fillna(0.0).set_index("elapsed_time").cumsum()


# --------------------------------------------------------------------------
# Volume reconstruction (needed for dilution terms in mass balances)
# --------------------------------------------------------------------------
def volume_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Per sampling interval: measured delta V vs. total feed volume added.

    `removed` = sum(feeds) - delta V is the unaccounted volume loss
    (sampling and/or evaporation), about 0.03-0.05 per interval.
    """
    rows = []
    for b, g in df.groupby("batch_id", sort=False):
        v = g[g["volume"].notna()]
        for (_, a), (_, c) in zip(v.iterrows(), list(v.iterrows())[1:]):
            win = g[(g["elapsed_time"] > a["elapsed_time"]) & (g["elapsed_time"] <= c["elapsed_time"])]
            fed = win[FEEDS].sum().sum()
            rows.append(dict(batch_id=b, t_start=a["elapsed_time"], t_end=c["elapsed_time"],
                             dV=c["volume"] - a["volume"], fed=fed,
                             removed=fed - (c["volume"] - a["volume"])))
    return pd.DataFrame(rows)


def volume_profile(df: pd.DataFrame, batch_id: str) -> pd.DataFrame:
    """Piecewise-constant reactor volume at every event of one batch.

    Starts from each measured volume and adds feed boluses as they happen;
    at the next sampling time the volume is reset to the measured value
    (the step absorbs sampling/evaporation losses). Works for test batches
    too, since volume is reported on the to_predict rows.
    """
    g = df[df["batch_id"] == batch_id].sort_values("elapsed_time")
    out, v = [], np.nan
    for _, r in g.iterrows():
        if not np.isnan(r["volume"]):
            v = r["volume"]
        else:
            v = v + np.nansum(r[FEEDS].to_numpy(dtype=float))
        out.append((r["elapsed_time"], v, r["row_type"]))
    return pd.DataFrame(out, columns=["elapsed_time", "volume", "row_type"])


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def batch_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per batch: split, duration, sample count, feed counts/totals, initial state."""
    m = measurements(df, include_to_predict=True)
    f = feeds_long(df)
    test = set(test_batches(df))

    g = m.groupby("batch_id")
    s = pd.DataFrame({
        "split": ["test" if b in test else "train" for b in g.size().index],
        "n_samples": g.size(),
        "t_end": g["elapsed_time"].max(),
        "V0": g["volume"].first(),
        "V_end": g["volume"].last(),
    })
    init = m[m["elapsed_time"] == 0].set_index("batch_id")[TARGETS].add_suffix("_0")
    n = f.pivot_table(index="batch_id", columns="feed", values="amount", aggfunc="count").add_prefix("n_")
    tot = f.pivot_table(index="batch_id", columns="feed", values="amount", aggfunc="sum").add_prefix("tot_")
    s = s.join(init).join(n).join(tot)
    return s.loc[_natural(s.index)]

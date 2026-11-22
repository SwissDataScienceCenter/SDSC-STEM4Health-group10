"""Simple reference models (no mechanism, ignore feeds).

(a) MeanTrajectory: average training curve per culture day, linearly interpolated at the requested times.
(b) MeanTrajectory(scale_cells=True): same, with cell densities rescaled by the batch's initial VCD
    relative to the training mean initial VCD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data as D

VARS = ["vcd", "dcd", "tcd", "substrate", "byproduct"]


class MeanTrajectory:
    def __init__(self, scale_cells: bool = False):
        self.scale_cells = scale_cells

    def fit(self, raw: pd.DataFrame, train_ids: list[str]):
        m = D.measurements(raw[raw.batch_id.isin(train_ids)])
        m["day"] = m["elapsed_time"].round().astype(int)
        self.curve = m.groupby("day")[["elapsed_time", *VARS]].mean()   # mean time and values per day
        self.vcd0 = m.loc[m.elapsed_time == 0, "vcd"].mean()
        return self

    def predict(self, raw: pd.DataFrame, batch_ids: list[str]) -> pd.DataFrame:
        m = D.measurements(raw[raw.batch_id.isin(batch_ids)], include_to_predict=True)
        x0 = m[m.elapsed_time == 0].set_index("batch_id")["vcd"]
        out = m.loc[m.elapsed_time > 0, ["batch_id", "elapsed_time"]].copy()
        for v in VARS:
            out[v] = np.interp(out["elapsed_time"], self.curve["elapsed_time"], self.curve[v])
            if self.scale_cells and v in ("vcd", "dcd", "tcd"):
                out[v] *= out["batch_id"].map(x0).to_numpy() / self.vcd0
        return out.reset_index(drop=True)

"""Experiment: extended features (no Feed C / Feed D). Groups G1..G4 added to the reference per-target lists."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as X  # noqa: E402

GROUPS = {
    "G1": ["fb_frac_t", "fs_frac_t", "dilution", "log_Vratio", "added_S_conc_t"],
    "G2": ["t_since_fs", "fb_recent_2d", "fb_first_half", "fb_second_half", "n_fs_t"],
    "G3": ["log_t", "log1p_fb_cum", "log1p_fs_cum"],
    "G4": ["log_viab0", "log_sub_by0", "log_dcd0"],
}


def build_extended_table(raw, table):
    f = X.D.feeds_long(raw)
    fg = {b: g for b, g in f.groupby("batch_id")}
    t = table.copy()
    new = {k: [] for k in sum(GROUPS.values(), [])}
    for r in t.itertuples():
        g = fg[r.batch_id]
        B = g[g.feed == "feed_b"]
        S = g[g.feed == "feed_substrate"]
        tt, Vt = r.t, r.Vt
        Bu, Su = B[B.elapsed_time <= tt], S[S.elapsed_time <= tt]
        new["fb_frac_t"].append(r.feed_b_cum_t / Vt)
        new["fs_frac_t"].append(r.feed_substrate_cum_t / Vt)
        new["dilution"].append(r.V0 / Vt)
        new["log_Vratio"].append(np.log(Vt / r.V0))
        new["added_S_conc_t"].append((50 * Bu.amount.sum() + 250 * Su.amount.sum()) / Vt)
        new["t_since_fs"].append(tt - Su.elapsed_time.max() if len(Su) else 20.0)
        new["fb_recent_2d"].append(B[(B.elapsed_time <= tt) & (B.elapsed_time > tt - 2)].amount.sum())
        mid = B.elapsed_time.median()
        new["fb_first_half"].append(B[B.elapsed_time <= mid].amount.sum())
        new["fb_second_half"].append(B[B.elapsed_time > mid].amount.sum())
        new["n_fs_t"].append(len(Su))
        new["log_t"].append(np.log(tt))
        new["log1p_fb_cum"].append(np.log1p(r.feed_b_cum_t))
        new["log1p_fs_cum"].append(np.log1p(r.feed_substrate_cum_t))
        new["log_viab0"].append(r.log_vcd0 - r.log_tcd0)
        new["log_sub_by0"].append(r.log_substrate0 - r.log_byproduct0)
        new["log_dcd0"].append(np.log(max(np.exp(r.log_tcd0) - np.exp(r.log_vcd0), 1e-3)))
    return t.assign(**new)


def main():
    raw, table, train, test = X.load()
    tab = build_extended_table(raw, table)
    ref = X.R.FEATURES_BY_TARGET
    res = {}
    def run(name, feats):
        res[name] = X.report(name, X.lobo(lambda tr, te: X.ridge_blend(tr, te, features=feats, tab=tab)))
    for g, cols in GROUPS.items():
        run(f"feat_{g}", {v: ref[v] + cols for v in X.TARGETS})
    refs = pd.read_csv(X.OUT / "ref_ridge_blend_score.csv", index_col=0).iloc[:, 0] if (X.OUT / "ref_ridge_blend_score.csv").exists() else pd.Series({"vcd": 6.892, "tcd": 6.858, "substrate": 7.042, "byproduct": 9.104})
    best = {}
    for v in X.TARGETS:
        best[v] = list(ref[v])
        for g, cols in GROUPS.items():
            if refs[v] - res[f"feat_{g}"][v] > 0.05:
                best[v] += cols
        print(v, "uses", [g for g in GROUPS if best[v] != ref[v] and set(GROUPS[g]) <= set(best[v])])
    run("feat_best", best)
    rm = np.mean([refs[v] for v in X.TARGETS])
    print("ref mean", rm, "best mean", res["feat_best"]["mean"])
    if rm - res["feat_best"]["mean"] > 0.1:
        X.write_submission("feat_best", X.ridge_blend(train, test, features=best, tab=tab))
        (X.OUT / "feat_best_features.json").write_text(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()

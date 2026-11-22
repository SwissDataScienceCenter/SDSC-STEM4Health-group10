"""Final submission = blend_nonlinear (src/model_families.py) + substrate controller rules (src/control_rules.py):
  * mid-run extra-bolus rule (SubstrateControlRule), re-applied on the FINAL blend (it was only applied to our model,
    i.e. diluted to 50 % in the team blend): substrate MAPE -0.048 (CV seeds 0-2) / -0.057 (fresh seeds 3-5);
  * late dose rule (LateDoseRule, samples 11-12): -0.116 / -0.169 in CV, -0.29 on the leaderboard (8.297 -> 8.003).

Usage (repo root), after src/run_regression.py, src/blend_team.py and src/model_families.py:
    python src/final_submission.py           # -> submissions/blend_nonlinear_rules.csv (both rules)
    python src/final_submission.py --late-only  # -> submissions/blend_nonlinear_dose.csv (LB 7.613)
    python src/final_submission.py --base       # -> submissions/blend_base.csv (+ hidden-base byproduct model)
    python src/final_submission.py --base2      # -> submissions/blend_base2.csv (+ hidden base for substrate too)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import control_rules as C
import data as D

REPO = Path(__file__).resolve().parents[1]
T = ["vcd", "tcd", "substrate", "byproduct"]


def main():
    raw = D.load_raw()
    tpl = D.load_template()
    sub = pd.read_csv(REPO / "submissions" / "blend_nonlinear.csv")
    assert (sub.batch_id.to_numpy() == tpl.batch_id.to_numpy()).all() and np.allclose(sub.elapsed_time, tpl.elapsed_time)
    samp = D.measurements(raw, include_to_predict=True)
    samp["j"] = samp.groupby("batch_id").cumcount()
    k = lambda d: d.assign(_t=d["elapsed_time"].round(6))
    sub = k(sub).merge(k(samp)[["batch_id", "_t", "j"]], on=["batch_id", "_t"], how="left")
    assert sub.j.notna().all()
    late_only = "--late-only" in sys.argv
    train = D.train_batches(raw)
    out = sub if late_only else C.SubstrateControlRule().fit(raw, train).apply(sub)
    rule = C.LateDoseRule().fit(raw, train)
    out = rule.apply(out)
    changed = (out.substrate != sub.substrate).sum()
    out = out.drop(columns=["_t", "j"])
    assert len(out) == len(tpl) and out[T].notna().all().all() and (out[T] > 0).all().all()
    name = "blend_nonlinear_dose.csv" if late_only else "blend_nonlinear_rules.csv"
    out.to_csv(REPO / "submissions" / name, index=False)
    print(f"late-dose rule: cap {rule.cap:.3f}, floor {rule.floor:.3f}; {changed} substrate predictions changed")
    print(f"submission written: submissions/{name}")




def add_base_byproduct():
    """submissions/blend_base.csv = blend_nonlinear_dose.csv with byproduct = 0.9 hidden-base-volume Huber model
    (src/byproduct_base.py) + 0.1 teammate model (geometric)."""
    import byproduct_base as BP
    raw = D.load_raw()
    sub = pd.read_csv(REPO / "submissions" / "blend_nonlinear_dose.csv")
    pt = BP.predict(BP.build_table(raw), D.train_batches(raw), D.test_batches(raw))
    k = lambda d: d.assign(_t=d["elapsed_time"].round(6))
    pt = k(sub)[["batch_id", "_t"]].merge(k(pt)[["batch_id", "_t", "byproduct"]], on=["batch_id", "_t"], how="left")
    assert pt.byproduct.notna().all() and len(pt) == len(sub)
    out = sub.copy()
    out["byproduct"] = np.exp(BP.W_BLEND * np.log(pt.byproduct.to_numpy()) + (1 - BP.W_BLEND) * np.log(sub.byproduct.to_numpy()))
    assert out[T].notna().all().all() and (out[T] > 0).all().all()
    out.to_csv(REPO / "submissions" / "blend_base.csv", index=False)
    print("submission written: submissions/blend_base.csv")

# Weights of the "base" version (CV, same folds, chosen on seeds 0-2, checked on fresh seeds 3-5):
#   substrate : 0.75 ours_base + 0.25 teammate, then late dose rule   6.70 / 6.81 -> 6.48 / 6.58
#   byproduct : 0.7 note model + 0.2 ours_base + 0.1 teammate         8.67 -> 7.87 / 7.80
# ours_base = our hybrid_ensemble with base_production added to the substrate and byproduct features.
W_S_OURS_BASE = 0.75
W_P = {"note": 0.7, "ours_base": 0.2, "teammate": 0.1}


def add_base_all():
    """submissions/blend_base2.csv: vcd / tcd from blend_nonlinear_dose.csv; substrate and byproduct rebuilt with
    the hidden base volume (src/base_volume.py)."""
    import base_volume as BV
    import byproduct_base as BP
    import regression as R
    from run_regression import predict
    raw = D.load_raw()
    train, test = D.train_batches(raw), D.test_batches(raw)
    sub = pd.read_csv(REPO / "submissions" / "blend_nonlinear_dose.csv")
    k = lambda d: d.assign(_t=d["elapsed_time"].round(6))
    align = lambda p, cols: k(sub)[["batch_id", "_t"]].merge(k(p)[["batch_id", "_t", *cols]], on=["batch_id", "_t"], how="left")
    tab = BV.add_base_features(R.build_table(raw), raw)
    feats = dict(R.FEATURES_BY_TARGET)
    for v in ("substrate", "byproduct"):
        feats[v] = feats[v] + ["base_production"]
    ob = align(predict(raw, tab, feats, train, test, 0.2, R.N_EIGENCURVES, ensemble=True, rules=True, ode=True),
               ["substrate", "byproduct", "j"])
    th = align(pd.read_csv(REPO / "src" / "submission.csv"), ["substrate", "byproduct"])        # teammate (src/v2.py)
    note = align(BP.predict(BP.build_table(raw), train, test), ["byproduct"])
    for d in (ob, th, note):
        assert d.notna().all().all() and len(d) == len(sub)
    out = sub.copy()
    lg = np.log
    out["byproduct"] = np.exp(W_P["note"] * lg(note.byproduct) + W_P["ours_base"] * lg(ob.byproduct)
                              + W_P["teammate"] * lg(th.byproduct)).to_numpy()
    s = pd.DataFrame({"batch_id": sub.batch_id, "j": ob.j.astype(int),
                      "substrate": np.exp(W_S_OURS_BASE * lg(ob.substrate) + (1 - W_S_OURS_BASE) * lg(th.substrate))})
    out["substrate"] = C.LateDoseRule().fit(raw, train).apply(s).substrate.to_numpy()
    assert out[T].notna().all().all() and (out[T] > 0).all().all()
    out.to_csv(REPO / "submissions" / "blend_base2.csv", index=False)
    ch = lambda v: (100 * (out[v] - sub[v]).abs() / sub[v]).median().round(2)
    print(f"submission written: submissions/blend_base2.csv | median change vs blend_nonlinear_dose: substrate {ch('substrate')} %, byproduct {ch('byproduct')} %")


if __name__ == "__main__":
    if "--base2" in sys.argv:
        add_base_all()
    elif "--base" in sys.argv:
        add_base_byproduct()
    else:
        main()

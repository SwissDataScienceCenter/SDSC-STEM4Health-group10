"""Final submission = blend_nonlinear (src/model_families.py) + substrate controller rules (src/control_rules.py):
  * mid-run extra-bolus rule (SubstrateControlRule), re-applied on the FINAL blend (it was only applied to our model,
    i.e. diluted to 50 % in the team blend): substrate MAPE -0.048 (CV seeds 0-2) / -0.057 (fresh seeds 3-5);
  * late dose rule (LateDoseRule, samples 11-12): -0.116 / -0.169 in CV, -0.29 on the leaderboard (8.297 -> 8.003).

Usage (repo root), after src/run_regression.py, src/blend_team.py and src/model_families.py:
    python src/final_submission.py           # -> submissions/blend_nonlinear_rules.csv (both rules)
    python src/final_submission.py --late-only  # -> submissions/blend_nonlinear_dose.csv (LB 7.613)
    python src/final_submission.py --base       # -> submissions/blend_base.csv (+ hidden-base byproduct model)
    python src/final_submission.py --base2      # -> submissions/blend_base2.csv (+ hidden base for substrate too)
    python src/final_submission.py --base3      # -> submissions/blend_base3.csv (refined base features + re-chosen blends)
    python src/final_submission.py --kalman     # -> submissions/blend_base_kalman.csv (blend_base + Feed B controller update, LB 7.438)
    python src/final_submission.py --kalman2    # -> submissions/blend_base_kalman2.csv (tuned update: days 1,2,4-8,11, shrink 0.4; LB 7.410)
    python src/final_submission.py --kalman3 [--mid]  # -> blend_base_kalman3_late.csv / _full.csv (control-rule constraints inside the update; full: LB 7.300)
    python src/final_submission.py --kalman4    # -> blend_kalman4.csv (re-tuned constrained update + byproduct with mid-run indicator)
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


# Weights of the "base3" version (CV, same folds, chosen on seeds 0-2, checked on fresh seeds 3-5):
#   substrate : 0.75 ours_basec + 0.25 teammate, then late dose rule         6.70 / 6.81 (blend_base) -> 6.35 / 6.45
#   byproduct : 0.4 refined note model + 0.4 note model + 0.1 ours_basec + 0.1 teammate     7.90 / 7.83 -> 7.86 / 7.79
# ours_basec = our hybrid_ensemble with the refined base features (substrate: base_c2345, byproduct: base_c345).
W3_S = {"ours_basec": 0.75, "teammate": 0.25}
W3_P = {"note_refined": 0.4, "note": 0.4, "ours_basec": 0.1, "teammate": 0.1}


def add_base3():
    import base_volume as BV
    import byproduct_base as BP
    import regression as R
    from run_regression import predict
    raw = D.load_raw()
    train, test = D.train_batches(raw), D.test_batches(raw)
    sub = pd.read_csv(REPO / "submissions" / "blend_nonlinear_dose.csv")
    k = lambda d: d.assign(_t=d["elapsed_time"].round(6))
    align = lambda p, cols: k(sub)[["batch_id", "_t"]].merge(k(p)[["batch_id", "_t", *cols]], on=["batch_id", "_t"], how="left")
    tab = BV.add_refined_base_features(R.build_table(raw), raw)
    feats = dict(R.FEATURES_BY_TARGET)
    feats["substrate"] = feats["substrate"] + ["base_c2345"]
    feats["byproduct"] = feats["byproduct"] + ["base_c345"]
    obc = align(predict(raw, tab, feats, train, test, 0.2, R.N_EIGENCURVES, ensemble=True, rules=True, ode=True),
                ["substrate", "byproduct", "j"])
    th = align(pd.read_csv(REPO / "src" / "submission.csv"), ["substrate", "byproduct"])       # teammate (src/v2.py)
    btab = BV.add_refined_base_features(BP.build_table(raw), raw)
    note = align(BP.predict(btab, train, test), ["byproduct"])
    note_c = align(BP.predict(btab, train, test, **BP.REFINED), ["byproduct"])
    for d in (obc, th, note, note_c):
        assert d.notna().all().all() and len(d) == len(sub)
    lg = np.log
    out = sub.copy()
    out["byproduct"] = np.exp(W3_P["note_refined"] * lg(note_c.byproduct) + W3_P["note"] * lg(note.byproduct)
                              + W3_P["ours_basec"] * lg(obc.byproduct) + W3_P["teammate"] * lg(th.byproduct)).to_numpy()
    s = pd.DataFrame({"batch_id": sub.batch_id, "j": obc.j.astype(int),
                      "substrate": np.exp(W3_S["ours_basec"] * lg(obc.substrate) + W3_S["teammate"] * lg(th.substrate))})
    out["substrate"] = C.LateDoseRule().fit(raw, train).apply(s).substrate.to_numpy()
    assert out[T].notna().all().all() and (out[T] > 0).all().all()
    out.to_csv(REPO / "submissions" / "blend_base3.csv", index=False)
    ref = pd.read_csv(REPO / "submissions" / "blend_base.csv")
    ch = lambda v: (100 * (out[v] - ref[v]).abs() / ref[v]).median().round(2)
    print(f"submission written: submissions/blend_base3.csv | median change vs blend_base (LB 7.464): substrate {ch('substrate')} %, byproduct {ch('byproduct')} %")


def add_kalman(tuned=False, constraints=None, v4=False):
    """submissions/blend_base_kalman.csv = blend_base.csv (LB 7.464) with the substrate updated by the Feed B
    controller (src/substrate_kalman.py), before the late dose rule (as validated in CV)."""
    import substrate_kalman as SK
    raw = D.load_raw()
    train = D.train_batches(raw)
    sub = pd.read_csv(REPO / "submissions" / "blend_base.csv")
    k = lambda d: d.assign(_t=d["elapsed_time"].round(6))
    align = lambda p: k(sub)[["batch_id", "_t"]].merge(k(p)[["batch_id", "_t", "substrate"]], on=["batch_id", "_t"], how="left").substrate.to_numpy()
    ours, th = align(pd.read_csv(REPO / "submissions" / "hybrid_ensemble.csv")), align(pd.read_csv(REPO / "src" / "submission.csv"))
    pre = np.exp(0.5 * np.log(ours) + 0.5 * np.log(th))                     # blend_team substrate (before late rule)
    samp = D.measurements(raw, include_to_predict=True)
    samp["j"] = samp.groupby("batch_id").cumcount()
    j = k(sub).merge(k(samp)[["batch_id", "_t", "j"]], on=["batch_id", "_t"], how="left").j.astype(int).to_numpy()
    late = C.LateDoseRule().fit(raw, train)
    late_c = C.LateDoseRule(SK.V4["late_q"]).fit(raw, train) if v4 else late
    chk = late.apply(pd.DataFrame({"batch_id": sub.batch_id, "j": j, "substrate": pre})).substrate.to_numpy()
    assert np.allclose(chk, sub.substrate), "pre-rule substrate does not reproduce blend_base"
    # prior errors: out-of-fold relative errors of the same blend on the training batches (6 CV seeds pooled)
    OUTF = REPO / "results" / "families"
    rel = []
    for s in range(6):
        ou = pd.read_csv(OUTF / f"ours_oof_s{s}.csv").set_index(["batch_id", "j"])
        te = pd.read_csv(OUTF / f"their_oof_s{s}.csv").set_index(["batch_id", "j"]).loc[ou.index]
        P = np.exp(0.5 * np.log(ou.substrate) + 0.5 * np.log(te.substrate)).unstack()[list(range(1, 14))]
        rel.append((P - SK.ControllerUpdate(raw).S.loc[P.index, list(range(1, 14))]) / SK.ControllerUpdate(raw).S.loc[P.index, list(range(1, 14))])
    cu = SK.ControllerUpdate(raw).fit(train, pd.concat(rel), **(dict(days=SK.V4["days"], shrink=SK.V4["shrink"]) if v4 else SK.TUNED if tuned else {}))
    new = pre.copy()
    for b in sub.batch_id.unique():
        m = (sub.batch_id == b).to_numpy()
        order = np.argsort(j[m])
        mu = pre[m][order]
        assert (j[m][order] == np.arange(1, 14)).all()
        cons = {}
        if constraints in ("late", "full"):
            cons["late"] = late_c
        if constraints == "full":
            cons["mid"] = C.SubstrateControlRule().fit(raw, train)
        if v4:
            cons.update(noise_scale=SK.V4["noise_scale"], tau_mid=SK.V4["tau_mid"])
        upd = cu.update(b, mu, **cons)
        new_b = np.empty(13); new_b[order] = upd
        new[m] = new_b
    out = sub.copy()
    out["substrate"] = late_c.apply(pd.DataFrame({"batch_id": sub.batch_id, "j": j, "substrate": new})).substrate.to_numpy()
    if v4:   # byproduct: hidden-base Huber model + mid-run bolus indicator (src/byproduct_base.py TRIG), 0.9 + 0.1 teammate
        import byproduct_base as BP
        btab = BP.add_trigger(BP.build_table(raw), raw)
        kb = lambda d: d.assign(_t=d["elapsed_time"].round(6))
        pt = kb(sub)[["batch_id", "_t"]].merge(kb(BP.predict(btab, train, D.test_batches(raw), **BP.TRIG))[["batch_id", "_t", "byproduct"]],
                                                on=["batch_id", "_t"], how="left").byproduct.to_numpy()
        tb = kb(sub)[["batch_id", "_t"]].merge(kb(pd.read_csv(REPO / "src" / "submission.csv"))[["batch_id", "_t", "byproduct"]],
                                                on=["batch_id", "_t"], how="left").byproduct.to_numpy()
        out["byproduct"] = np.exp(0.9 * np.log(pt) + 0.1 * np.log(tb))
    assert out[T].notna().all().all() and (out[T] > 0).all().all()
    name = "blend_kalman4.csv" if v4 else {None: "blend_base_kalman2.csv" if tuned else "blend_base_kalman.csv",
                                           "late": "blend_base_kalman3_late.csv", "full": "blend_base_kalman3_full.csv"}[constraints]
    out.to_csv(REPO / "submissions" / name, index=False)
    ch = 100 * (out.substrate - sub.substrate) / sub.substrate
    print(f"submission written: submissions/{name} | substrate change vs blend_base: median |.| {ch.abs().median():.2f} %, max |.| {ch.abs().max():.1f} %")
    print("mean change per day (%):", ch.groupby(j).mean().round(2).to_dict())


if __name__ == "__main__":
    if "--kalman4" in sys.argv:
        add_kalman(tuned=True, constraints="full", v4=True)
    elif "--kalman3" in sys.argv:
        add_kalman(tuned=True, constraints="full" if "--mid" in sys.argv else "late")
    elif "--kalman2" in sys.argv:
        add_kalman(tuned=True)
    elif "--kalman" in sys.argv:
        add_kalman()
    elif "--base3" in sys.argv:
        add_base3()
    elif "--base2" in sys.argv:
        add_base_all()
    elif "--base" in sys.argv:
        add_base_byproduct()
    else:
        main()

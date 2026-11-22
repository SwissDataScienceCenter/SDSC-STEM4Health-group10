"""Neural ODE for the SDSC STEM4Health fed-batch CHO challenge.

Follows Chiu & Du (2025), "A neural ODE model for predicting the growth of
CHO cells in a bioreactor" (paper_CHO.pdf):
  - small fully-connected NN (2 hidden layers, Tanh then ReLU) replaces the
    kinetics: dy/dt = NN(y, t, batch features, feed rates),
  - observations standardized before training,
  - loss = per-batch MAE, averaged over batches (their Eqs. 20-21), plus a
    relative-error term because the leaderboard scores MAPE,
  - adaptive RK solver (Dormand-Prince), irregular observation times handled
    natively by the ODE formulation,
  - trained jointly across all batches so it extrapolates to unseen initial
    conditions / feed schedules (their testing design).

Adaptations for this dataset (fed-batch, unlike the paper's batch runs):
  - feed events are exact impulses: volume V is a state with dV/dt = feed
    rate; concentrations dilute by r/V; substrate gains c_ss*r_fs + c_sb*r_fb
    with learnable feed concentrations c_ss, c_sb,
  - state is [Xv, Xd, S, P, V]; targets vcd=Xv, tcd=Xv+Xd, S, P (the paper
    models only Xv/GLC/LAC; tcd requires the dead-cell balance).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]

T_END = 13.4      # simulation horizon (days); data ends ~13.0
DT = 0.05         # solver step / grid spacing (days); == EPS so impulses align with cells
EPS = 0.05        # window over which a feed impulse is spread into a rate
R_SCALE = 10.0    # scaling of feed-rate inputs to the NN
V_SCALE = 20.0

DATA_CANDIDATES = [
    "/home/renku/work/input-data/Project 1 - DTB",
    os.path.join(os.path.dirname(__file__), "..", "..", "input-data", "Project 1 - DTB"),
    "./data",
    ".",
]


def find_data_dir():
    for c in DATA_CANDIDATES:
        if os.path.exists(os.path.join(c, "bioreaction_remaining.csv")):
            return os.path.abspath(c)
    raise FileNotFoundError("bioreaction_remaining.csv not found in " + str(DATA_CANDIDATES))


def load_data(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "bioreaction_remaining.csv"))
    tpl = pd.read_csv(os.path.join(data_dir, "bioreaction_template.csv"), dtype={"batch_id": str})
    batches = {}
    for b, g in df.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        obs = g[g.vcd.notna()]
        feeds = g[g[FEEDS].notna().any(axis=1)]
        y0row = obs.iloc[0]
        batches[str(b)] = dict(
            t0=float(y0row.elapsed_time),
            y0=np.array([y0row.vcd, y0row.tcd - y0row.vcd,
                         y0row.substrate, y0row.byproduct], float),
            V0=float(y0row.volume),
            f_t=feeds.elapsed_time.values.astype(float),
            f_a=feeds[FEEDS].fillna(0.0).values.astype(float),
            t_obs=obs.elapsed_time.values.astype(float) if len(obs) > 1 else None,
            obs=obs[TARGETS].values.astype(float) if len(obs) > 1 else None,
            to_predict=bool(g.to_predict.any()),
            volume_log=g[["elapsed_time", "volume"]].dropna().sort_values("elapsed_time"),
        )
    return batches, tpl, df


def static_features(bd):
    f_t, f_a = bd["f_t"], bd["f_a"]
    m = f_t > bd["t0"] + 1e-9
    ft, fa = f_t[m], f_a[m]
    tot = fa.sum(0) if len(fa) else np.zeros(4)
    v_end = float(bd["volume_log"].volume.iloc[-1])
    dts = np.diff(np.sort(np.concatenate([[bd["t0"]], ft]))) if len(ft) else np.array([0.0])
    return np.array([
        bd["y0"][0], bd["y0"][1], bd["y0"][2], bd["y0"][3], bd["V0"],
        len(ft), tot.sum(), tot[0], tot[1], tot[2], tot[3],
        (ft[0] if len(ft) else bd["t0"]), float(np.mean(dts)) if len(dts) else 0.0,
        v_end,
    ], float)


def feed_rate_grid(bd, t_grid):
    """Feed amounts as rates on the grid: impulse f at t spread over EPS.

    Returns (T, 7): [r_fs, r_fb, r_fc, r_fd, cum_total, cum_fs, cum_fb] where the
    cumulative columns are the fed volumes up to the start of each cell."""
    R = np.zeros((len(t_grid), 4))
    n_cells = max(1, int(np.ceil(EPS / DT)))
    for t, a in zip(bd["f_t"], bd["f_a"]):
        if t <= bd["t0"] + 1e-9 or t >= t_grid[-1]:
            continue
        k = int(t // DT)
        for j in range(n_cells):
            kk = k + j
            if kk >= len(R):
                break
            R[kk] += a / (n_cells * DT)
    C = np.cumsum(R, axis=0) * DT          # cumulative fed volume per feed type
    return np.concatenate([R, C[:, :2].sum(1, keepdims=True), C[:, :2]], axis=1)


class NeuralODE(nn.Module):
    """dy/dt = physics(feed impulses, dilution) + NN(kinetics), per paper.

    The NN also sees the cohort-median target trajectory at time t, so it
    learns per-batch deviations around the population average (the cohort is
    homogeneous; this is what makes feed-bump dynamics learnable)."""

    def __init__(self, n_static, sd, U_all, t_grid, med, hidden=96):
        super().__init__()
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))       # (4,)
        self.register_buffer("U_all", torch.tensor(U_all, dtype=torch.float32))  # (B, T, 7)
        self.register_buffer("t_grid", torch.tensor(t_grid, dtype=torch.float32))
        self.register_buffer("med", torch.tensor(med, dtype=torch.float32))      # (T, 4)
        self.c_feed = nn.Parameter(torch.tensor([100.0, 8.0, 0.0, 0.0]))  # substrate conc per feed (g/L)
        d_in = 4 + 1 + 1 + n_static + 7 + 4
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )
        self.n_static = n_static

    def rhs(self, t, y):
        # y: (B, 5) = [Xv, Xd, S, P, V]
        B = y.shape[0]
        k = int(torch.clamp((t / DT).long(), 0, self.U_all.shape[1] - 1))
        U = self.U_all[:, k, :]                                   # (B, 7) rates + cumulative feeds
        r = U[:, :4]
        ys = y[:, :4] / self.sd
        vs = y[:, 4:5] / V_SCALE
        t_in = (t / T_END).expand(B, 1)
        med_in = (self.med[k] / self.sd).expand(B, 4)
        inp = torch.cat([ys, vs, t_in, self.statics, U[:, :4] / R_SCALE, U[:, 4:] / 7.0, med_in], dim=1)
        nn_out = self.net(inp)                                    # scaled conc derivatives
        rtot = r.sum(dim=1, keepdim=True)                         # (B, 1)
        V = y[:, 4:5].clamp(min=1.0)
        dil = y[:, :4] * rtot / V                                 # dilution of concentrations
        src = ((self.c_feed * r).sum(dim=1, keepdim=True)) / V    # substrate in feeds, (B,1)
        dconc = self.sd * nn_out - dil
        dconc = torch.cat([dconc[:, :2], (dconc[:, 2:3] + src), dconc[:, 3:4]], dim=1)
        return torch.cat([dconc, rtot], dim=1)

    def forward(self, t, y):
        return self.rhs(t, y)


def interp_at(sol, t_grid, times):
    """sol: (T, B, D) -> (B, P, D) at per-batch times (P,)."""
    T = t_grid.shape[0]
    idx = torch.searchsorted(t_grid, times).clamp(1, T - 1)
    t0, t1 = t_grid[idx - 1], t_grid[idx]
    w = ((times - t0) / (t1 - t0)).clamp(0, 1)
    out = sol[idx - 1] * (1 - w)[:, None, None] + sol[idx] * w[:, None, None]  # (P, B, D)
    return out.permute(1, 0, 2)


def targets_of(y):
    """(B, P, 5) state -> (B, P, 4) targets [vcd, tcd, S, P]."""
    return torch.cat([y[..., :1], y[..., :2].sum(-1, keepdim=True), y[..., 2:4]], dim=-1)


def build_tensors(batches, batch_ids, t_grid):
    y0 = np.stack([np.r_[batches[b]["y0"], batches[b]["V0"]] for b in batch_ids]).astype(np.float32)
    U = np.stack([feed_rate_grid(batches[b], t_grid) for b in batch_ids]).astype(np.float32)
    X = np.stack([static_features(batches[b]) for b in batch_ids]).astype(np.float32)
    return y0, U, X


def mape(pred, true):
    return float((np.abs(pred - true) / np.abs(true)).mean() * 100)


def build_context(batches, t_grid, train_ids):
    all_ids = sorted(batches.keys(), key=lambda s: int(s[5:]))
    y0_all, U_all, X_all = build_tensors(batches, all_ids, t_grid)
    obs_all = np.concatenate([batches[b]["obs"] for b in train_ids
                              if batches[b]["obs"] is not None])
    sd = obs_all.std(0) + 1e-6                      # target standardization (paper)
    mu, xs = X_all.mean(0), X_all.std(0) + 1e-6     # features standardized over ALL batches
    Xn = (X_all - mu) / xs
    # cohort-median target trajectory on the grid (train batches only), per obs index
    curves = []
    for b in train_ids:
        if batches[b]["obs"] is None:
            continue
        curves.append(np.stack([np.interp(t_grid, batches[b]["t_obs"], batches[b]["obs"][:, c])
                                for c in range(4)], axis=1))
    med = np.median(np.stack(curves), axis=0)       # (T, 4)
    return dict(all_ids=all_ids, y0_all=y0_all, U_all=U_all, Xn=Xn, sd=sd, mu=mu, xs=xs, med=med)


def train(train_ids, batches, t_grid, ctx, epochs=300, lr=3e-3, w_rel=2.0, w_neg=1.0,
          seed=0, verbose_every=25, eval_hook=None, eval_every=50, hidden=96):
    torch.manual_seed(seed)
    np.random.seed(seed)
    all_ids = ctx["all_ids"]
    pos = {b: i for i, b in enumerate(all_ids)}
    model = NeuralODE(n_static=ctx["Xn"].shape[1], sd=ctx["sd"],
                      U_all=ctx["U_all"], t_grid=t_grid, med=ctx["med"], hidden=hidden)
    tr = [pos[b] for b in train_ids]
    model.U_all = torch.tensor(ctx["U_all"][tr])
    model.statics = torch.tensor(ctx["Xn"][tr])
    y0 = torch.tensor(ctx["y0_all"][tr])

    obs_t = {b: (torch.tensor(batches[b]["t_obs"], dtype=torch.float32),
                 torch.tensor(batches[b]["obs"], dtype=torch.float32)) for b in train_ids}
    row_of = {b: i for i, b in enumerate(train_ids)}

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.5)
    tg = torch.tensor(t_grid, dtype=torch.float32)

    def loss_fn():
        sol = odeint_adjoint(model, y0, tg, method="rk4", options={"step_size": DT})
        tot, n = 0.0, 0
        for b in train_ids:
            times, obs = obs_t[b]
            pred = targets_of(interp_at(sol, tg, times))[row_of[b]]
            mae = ((pred - obs).abs() / model.sd).mean()          # paper Eq. 20-21 (std-MAE)
            rel = ((pred - obs).abs() / (obs.abs() + 0.1 * model.sd)).mean()  # MAPE-like
            neg = (torch.relu(-pred) ** 2).mean()
            tot = tot + mae + w_rel * rel + w_neg * neg
            n += 1
        return tot / n

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if ep % verbose_every == 0 or ep == 1:
            print(f"epoch {ep:4d}  loss {float(loss):.4f}", flush=True)
        if eval_hook is not None and (ep % eval_every == 0 or ep == epochs):
            eval_hook(ep, model)
    return model


def predict(model, batch_ids, ctx, t_grid, eval_times):
    """Simulate batches; eval_times: dict b -> np.array. Returns dict b -> (P,4) targets."""
    pos = {b: i for i, b in enumerate(ctx["all_ids"])}
    idx = [pos[b] for b in batch_ids]
    saved = (model.U_all, model.statics)
    model.U_all = torch.tensor(ctx["U_all"][idx])
    model.statics = torch.tensor(ctx["Xn"][idx])
    y0 = torch.tensor(ctx["y0_all"][idx])
    tg = torch.tensor(t_grid, dtype=torch.float32)
    with torch.no_grad():
        sol = odeint_adjoint(model, y0, tg, method="rk4", options={"step_size": DT})
        out = {}
        for j, b in enumerate(batch_ids):
            times = torch.tensor(eval_times[b], dtype=torch.float32)
            out[b] = np.maximum(targets_of(interp_at(sol, tg, times))[j].numpy(), 0.0)
    model.U_all, model.statics = saved
    return out


def holdout_mape(model, batches, ids, ctx, t_grid, med):
    eval_times = {b: batches[b]["t_obs"] for b in ids}
    preds = predict(model, ids, ctx, t_grid, eval_times)
    res = {}
    for c, tname in enumerate(TARGETS):
        m_model, m_base = [], []
        for b in ids:
            true = batches[b]["obs"][1:]
            m_model.append(np.abs(preds[b][1:] - true)[:, c] / np.abs(true)[:, c])
            m_base.append(np.abs(med[1:] - true)[:, c] / np.abs(true)[:, c])
        res[tname] = (100 * np.mean(m_model), 100 * np.mean(m_base))
    return res


def median_preds(batches, ids, ctx, t_grid):
    """Cohort-median baseline at each batch's obs times, via the med curve on the grid."""
    out = {}
    for b in ids:
        times = batches[b]["t_obs"]
        out[b] = np.stack([np.interp(times, t_grid, ctx["med"][:, c]) for c in range(4)], axis=1)
    return out


def blend_weights(batches, ids, ctx, t_grid, node_preds, base_preds, future_only=True):
    """Per-target shrinkage weight w minimizing holdout MAPE of w*node + (1-w)*base."""
    ws = {}
    for c, tname in enumerate(TARGETS):
        best_w, best = 1.0, np.inf
        for w in np.arange(0.0, 1.01, 0.1):
            apes = []
            for b in ids:
                sl = slice(1, None) if future_only else slice(None)
                true = batches[b]["obs"][sl, c]
                pr = w * node_preds[b][sl, c] + (1 - w) * base_preds[b][sl, c]
                apes.append(np.abs(pr - true) / np.abs(true))
            m = 100 * np.mean(apes)
            if m < best:
                best, best_w = m, w
        ws[tname] = (best_w, best)
    return ws


def main():
    import argparse
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--holdout-every", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--artifacts", default="/home/renku/work/team-workspace/neural_ode")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--no-blend", action="store_true")
    args = p.parse_args()

    data_dir = find_data_dir()
    print("data dir:", data_dir, flush=True)
    batches, tpl, df = load_data(data_dir)
    train_ids = sorted([b for b in batches if batches[b]["obs"] is not None
                        and len(batches[b]["obs"]) > 5], key=lambda s: int(s[5:]))
    pred_ids = sorted([b for b in batches if batches[b]["to_predict"]], key=lambda s: int(s[5:]))
    print(f"train batches: {len(train_ids)}, hidden batches: {len(pred_ids)}", flush=True)

    t_grid = np.arange(0.0, T_END + DT, DT)

    # ---------- validation: hold out every k-th train batch ----------
    holdout = train_ids[::args.holdout_every]
    tr = [b for b in train_ids if b not in holdout]
    print(f"holdout: {holdout}", flush=True)
    ctx_v = build_context(batches, t_grid, tr)      # med curve from train-only (no leakage)
    med = np.median(np.stack([batches[b]["obs"] for b in tr]), axis=0)

    def hook(ep, model):
        res = holdout_mape(model, batches, holdout, ctx_v, t_grid, med)
        line = "  ".join(f"{k}: node {v[0]:5.2f} / base {v[1]:5.2f}" for k, v in res.items())
        print(f"[eval ep {ep:4d}] {line}", flush=True)

    print("=== validation training (train minus holdout) ===", flush=True)
    model = train(tr, batches, t_grid, ctx_v, epochs=args.epochs, eval_hook=hook, eval_every=50)
    res = holdout_mape(model, batches, holdout, ctx_v, t_grid, med)
    print("\n=== holdout MAPE (%) on the 13 future points ===")
    for k, v in res.items():
        print(f"  {k:10s}  neuralODE {v[0]:5.2f}   median-baseline {v[1]:5.2f}")
    node_preds = predict(model, holdout, ctx_v, t_grid, {b: batches[b]["t_obs"] for b in holdout})
    base_preds = median_preds(batches, holdout, ctx_v, t_grid)
    ws = blend_weights(batches, holdout, ctx_v, t_grid, node_preds, base_preds)
    print("\n=== blended holdout MAPE (w*node + (1-w)*median, w fit on holdout) ===")
    for k, (w, m) in ws.items():
        print(f"  {k:10s}  w={w:.1f}   blended {m:5.2f}")
    mean_blend = float(np.mean([m for _, m in ws.values()]))
    mean_node = float(np.mean([v[0] for v in res.values()]))
    print(f"  mean: node {mean_node:.2f}  blended {mean_blend:.2f}")
    blend_w = {k: (1.0 if args.no_blend else w) for k, (w, _) in ws.items()}
    if args.validate_only:
        return

    # ---------- final: retrain on all train batches, predict hidden ----------
    print("\n=== final training on all train batches ===", flush=True)
    ctx = build_context(batches, t_grid, train_ids)
    model = train(train_ids, batches, t_grid, ctx, epochs=args.epochs, verbose_every=50)

    tpl_times = {b: tpl[tpl.batch_id == b].sort_values("elapsed_time").elapsed_time.values.astype(float)
                 for b in pred_ids}
    preds = predict(model, pred_ids, ctx, t_grid, tpl_times)
    base = {b: np.stack([np.interp(tpl_times[b], t_grid, ctx["med"][:, c]) for c in range(4)], axis=1)
            for b in pred_ids}

    rows = []
    for b in pred_ids:
        for k, t in enumerate(tpl_times[b]):
            row = {"batch_id": b, "elapsed_time": t}
            for c, cname in enumerate(TARGETS):
                row[cname] = blend_w[cname] * preds[b][k, c] + (1 - blend_w[cname]) * base[b][k, c]
            rows.append(row)
    fill = pd.DataFrame(rows)
    out = tpl.merge(fill, on=["batch_id", "elapsed_time"], how="left", suffixes=("", "_pred"))
    assert out[["vcd_pred", "tcd_pred", "substrate_pred", "byproduct_pred"]].notna().all().all()
    sub = out[["batch_id", "elapsed_time"]].copy()
    for c in TARGETS:
        sub[c] = np.maximum(out[c + "_pred"], 0.0)
    assert sub.notna().all().all() and np.isfinite(sub[TARGETS].values).all()

    os.makedirs(args.artifacts, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.artifacts, "model.pt"))
    with open(os.path.join(args.artifacts, "config.json"), "w") as f:
        json.dump({"sd": ctx["sd"].tolist(), "mu": ctx["mu"].tolist(),
                   "xs": ctx["xs"].tolist(), "epochs": args.epochs,
                   "blend_w": blend_w}, f)

    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nwrote {os.path.abspath(out_path)}  shape={sub.shape}  blend_w={blend_w}", flush=True)
    print(sub.head(3).to_string())


if __name__ == "__main__":
    main()

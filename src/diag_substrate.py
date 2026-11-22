import numpy as np
import torch
import neural_ode as no

torch.set_num_threads(1)
data_dir = no.find_data_dir()
batches, tpl, df = no.load_data(data_dir)
train_ids = sorted([b for b in batches if batches[b]["obs"] is not None and len(batches[b]["obs"]) > 5],
                   key=lambda s: int(s[5:]))
t_grid = np.arange(0.0, no.T_END + no.DT, no.DT)
ctx = no.build_context(batches, t_grid, train_ids)

holdout = train_ids[::8]
tr = [b for b in train_ids if b not in holdout]
print("holdout:", holdout, flush=True)
model = no.train(tr, batches, t_grid, ctx, epochs=250, verbose_every=50)
print("learned c_ss, c_sb:", float(model.c_ss), float(model.c_sb), flush=True)

preds = no.predict(model, holdout, ctx, t_grid, {b: batches[b]["t_obs"] for b in holdout})
for b in holdout[:3]:
    true = batches[b]["obs"]
    pred = preds[b]
    print(f"\n--- {b} (t, S_true, S_pred, P_true, P_pred, vcd_true, vcd_pred) ---")
    for k in range(len(true)):
        print(f"  {batches[b]['t_obs'][k]:5.2f}  S {true[k,2]:5.2f}/{pred[k,2]:5.2f}   "
              f"P {true[k,3]:5.2f}/{pred[k,3]:5.2f}   vcd {true[k,0]:5.2f}/{pred[k,0]:5.2f}")
    ape_S = np.abs(pred[1:, 2] - true[1:, 2]) / np.abs(true[1:, 2]) * 100
    print("  substrate APE% by point:", ape_S.round(1))

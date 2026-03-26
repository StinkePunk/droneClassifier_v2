# --- augment_sweep_main.py ---

import numpy as np
import torch
from datetime import datetime
import train_drone_cnn as tdc


P_APPLY_RIR    = [0.3, 0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
P_APPLY_ECHO   = [0.2, 0.2, 0.2, 0.1, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
P_APPLY_NOISE  = [0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.7, 0.6, 0.6, 0.6, 0.6]
P_PITCH_SHIFT  = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.3, 0.2, 0.2]
P_TIME_STRETCH = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.5]

SEEDS = [1,2,3,4,5,6,7,8,9,10,11]

with open("results.tsv", "w") as f:
    f.write("timestamp\tset_id\tseed\tcm00\tcm11\n")

for i in range(len(P_APPLY_RIR)):

    aug = {
        "P_APPLY_RIR": P_APPLY_RIR[i],
        "P_APPLY_ECHO": P_APPLY_ECHO[i],
        "P_APPLY_NOISE": P_APPLY_NOISE[i],
        "P_PITCH_SHIFT": P_PITCH_SHIFT[i],
        "P_TIME_STRETCH": P_TIME_STRETCH[i],
    }

    for seed in SEEDS:

        tdc.set_experiment_config(seed=seed, aug_probs=aug)
        cm_abs, model = tdc.train_model(return_outputs=True)

        # ONNX export: input width MUST match training feature shape (time_steps)
        time_steps = int(np.ceil(tdc.SEG_LEN / 512))  # hop_length=512 in extract_logmel
        torch.onnx.export(
            model,
            torch.randn(1, 1, 64, time_steps),
            f"model_set{i}_seed{seed}.onnx",
            opset_version=11
        )

        cm_abs = cm_abs[1]                 # <- array([[...]])
        cm_abs = cm_abs.astype(float)
        cm_percent = cm_abs / cm_abs.sum(axis=1, keepdims=True)

        with open("results.tsv", "a") as f:
            f.write(
                f"{datetime.now()}\t{i}\t{seed}\t"
                f"{cm_percent[0,0]:.4f}\t"
                f"{cm_percent[1,1]:.4f}\n"
            )
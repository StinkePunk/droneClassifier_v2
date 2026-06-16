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

SEEDS = [1,2,3,4,5,6,7]

with open("results.tsv", "w") as f:
    f.write("timestamp\tset_id\tseed\tcm00\tcm11\n")

for i in range(len(P_APPLY_RIR)):

    aug = {
        "apply_rir": P_APPLY_RIR[i],
            torch.randn(1, 3, tdc.IMG_SIZE, tdc.IMG_SIZE),
            f"model_set{i}_seed{seed}.onnx",
            opset_version=11
        )

        cm_abs = cm_abs.astype(float)
        cm_percent = cm_abs / cm_abs.sum(axis=1, keepdims=True)

        with open("results.tsv", "a") as f:
            f.write(
                f"{datetime.now()}\t{i}\t{seed}\t"
                f"{cm_percent[0,0]:.4f}\t"
                f"{cm_percent[1,1]:.4f}\n"
            )
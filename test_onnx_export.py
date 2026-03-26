import torch
import numpy as np
import train_drone_cnn as tdc

# Modell erzeugen (keine Gewichte nötig!)
model = tdc.ImprovedCNN()
model.eval()

# gleiche Input-Shape wie im Training
time_steps = int(np.ceil(tdc.SEG_LEN / 512))
dummy_input = torch.randn(1, 1, 64, time_steps)

# Export testen
torch.onnx.export(
    model,
    dummy_input,
    "test_model.onnx",
    opset_version=11
)

print("ONNX export erfolgreich!")
import torch
import numpy as np
from train import DragMLP

model = DragMLP()
model.load_state_dict(torch.load("drag_mlp.pt"))
model.eval()

# net[0] = Linear(4,32), net[1] = ReLU, net[2] = Linear(32,32), net[3]=ReLU, net[4]=Linear(32,1)
layer1 = model.net[0]  # Linear(4, 32)  -- small, not our target
layer2 = model.net[2]  # Linear(32, 32) -- our hand-tuning target
layer3 = model.net[4]  # Linear(32, 1)

# Save layer2's weight and bias as raw float32 binary, plus a known input/output pair
W2 = layer2.weight.detach().numpy().astype(np.float32)  # shape (32, 32)
b2 = layer2.bias.detach().numpy().astype(np.float32)    # shape (32,)

print("W2 shape:", W2.shape, "b2 shape:", b2.shape)

W2.tofile("../kernels/layer2_weight.bin")
b2.tofile("../kernels/layer2_bias.bin")

# Generate a test input (post layer1+relu, pre layer2) and PyTorch's reference output
rng = np.random.default_rng(7)
test_x = rng.normal(0, 1, size=(1, 32)).astype(np.float32)
test_x.tofile("../kernels/layer2_test_input.bin")

with torch.no_grad():
    ref_out = torch.relu(layer2(torch.tensor(test_x))).numpy()  # matmul + bias + relu

ref_out.tofile("../kernels/layer2_ref_output.bin")
print("Reference output:", ref_out)
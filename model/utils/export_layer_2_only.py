import torch
import torch.nn as nn
import numpy as np
from train import DragMLP

class Layer2Only(nn.Module):
    """Isolates just Linear(32,32) + ReLU -- the layer we hand-tuned in NEON."""
    def __init__(self, linear_layer):
        super().__init__()
        self.linear = linear_layer
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

def main():
    full_model = DragMLP()
    full_model.load_state_dict(torch.load("drag_mlp.pt"))
    full_model.eval()

    layer2 = full_model.net[2]  # the Linear(32,32) we've been targeting
    isolated = Layer2Only(layer2)
    isolated.eval()

    dummy_input = torch.randn(1, 32)

    torch.onnx.export(
        isolated,
        dummy_input,
        "layer2_only.onnx",
        input_names=["x"],
        output_names=["out"],
        opset_version=17,
        dynamo=False,
    )
    print("Exported layer2_only.onnx")

    # Sanity check: confirm this isolated model still matches the
    # same reference values we used for the NEON correctness tests
    test_x = np.fromfile("../kernels/layer2_test_input.bin", dtype=np.float32).reshape(1, 32)
    ref_out = np.fromfile("../kernels/layer2_ref_output.bin", dtype=np.float32).reshape(1, 32)

    with torch.no_grad():
        actual_out = isolated(torch.tensor(test_x)).numpy()

    max_diff = np.abs(actual_out - ref_out).max()
    print(f"Max diff vs kernels/ reference: {max_diff:.8f}")
    assert max_diff < 1e-4, "Isolated layer doesn't match the reference used in kernels/"
    print("PASSED: isolated layer matches the same reference used in your NEON tests.")

if __name__ == "__main__":
    main()
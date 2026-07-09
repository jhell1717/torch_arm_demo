import numpy as np
import torch
import onnxruntime as ort
from train import DragMLP

def main():
    # Load PyTorch model
    torch_model = DragMLP()
    torch_model.load_state_dict(torch.load("drag_mlp.pt"))
    torch_model.eval()

    # Load ONNX Runtime session
    session = ort.InferenceSession("drag_mlp.onnx")

    # Generate some test inputs (normalized scale, matching training distribution)
    rng = np.random.default_rng(123)
    test_inputs = rng.normal(0, 1, size=(20, 4)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_model(torch.tensor(test_inputs)).numpy()

    onnx_out = session.run(None, {"geometry_params": test_inputs})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"Max absolute difference: {max_diff:.8f}")

    tolerance = 1e-5
    assert max_diff < tolerance, f"Parity check FAILED: diff {max_diff} exceeds tolerance {tolerance}"
    print("PASSED: PyTorch and ONNX Runtime outputs match within tolerance.")

if __name__ == "__main__":
    main()
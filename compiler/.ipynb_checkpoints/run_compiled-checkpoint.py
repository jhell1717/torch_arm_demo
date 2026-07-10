import tvm
from tvm import relax
import numpy as np

dev = tvm.cpu()
ex = tvm.runtime.load_module("/work/compiler/drag_mlp_compiled.so")
vm = relax.VirtualMachine(ex, dev)

rng = np.random.default_rng(123)
test_input = rng.normal(0, 1, size=(1, 4)).astype(np.float32)

input_tensor = tvm.runtime.tensor(test_input, dev)
output = vm["main"](input_tensor)

print("TVM (Relax) compiled output:", output.numpy())

import onnxruntime as ort

session = ort.InferenceSession("/work/model/drag_mlp.onnx")
onnx_out = session.run(None, {"geometry_params": test_input})[0]

tvm_out = output.numpy()
max_diff = np.abs(tvm_out - onnx_out).max()
print(f"Max diff TVM (Relax) vs ONNX Runtime: {max_diff:.8f}")
assert max_diff < 1e-4, "Parity check failed"
print("PASSED")
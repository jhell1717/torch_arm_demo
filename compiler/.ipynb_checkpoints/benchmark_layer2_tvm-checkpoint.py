import tvm
from tvm import relax
import numpy as np
import time

dev = tvm.cpu()
ex = tvm.runtime.load_module("/work/compiler/layer2_only_compiled.so")
vm = relax.VirtualMachine(ex, dev)

test_x = np.fromfile("/work/kernels/layer2_test_input.bin", dtype=np.float32).reshape(1, 32)
ref_out = np.fromfile("/work/kernels/layer2_ref_output.bin", dtype=np.float32).reshape(1, 32)

input_tensor = tvm.runtime.tensor(test_x, dev)  # adjust if your TVM version used a different call earlier

# Correctness check first
output = vm["main"](input_tensor)
tvm_out = output.numpy()
max_diff = np.abs(tvm_out - ref_out).max()
print(f"TVM-compiled max diff vs reference: {max_diff:.8f}")
assert max_diff < 1e-4, "TVM output doesn't match reference"
print("PASSED: TVM-compiled isolated layer matches reference.\n")

# Benchmark
n_iters = 1_000_000
sink = 0.0

start = time.perf_counter()
for _ in range(n_iters):
    out = vm["main"](input_tensor)
    sink += float(out.numpy()[0, 0])  # forces the result to be used
end = time.perf_counter()

elapsed = end - start
ns_per_call = (elapsed / n_iters) * 1e9

print(f"TVM-compiled: {ns_per_call:.2f} ns/call")
print(f"(sink = {sink}, ignore, prevents dead-code elimination)")
import onnx
import tvm
from tvm.relax.frontend.onnx import from_onnx
import numpy as np

onnx_model = onnx.load("/work/model/drag_mlp.onnx")

# Shape must match what we used at export time: batch x 4 features
input_name = "geometry_params"
shape_dict = {input_name: (1, 4)}

mod = from_onnx(onnx_model, shape_dict=shape_dict)

# Since we're compiling *inside* a real AArch64 container, we target
# the native CPU directly rather than cross-compiling.
target = tvm.target.Target({"kind": "llvm", "mcpu": "native"})

with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relax.build(mod, target=target)

lib.export_library("/work/compiler/drag_mlp_compiled.so")
print("Compiled and saved drag_mlp_compiled.so")
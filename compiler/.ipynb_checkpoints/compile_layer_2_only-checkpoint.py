import onnx
import tvm
from tvm.relax.frontend.onnx import from_onnx
import numpy as np

onnx_model = onnx.load("/work/model/layer2_only.onnx")

shape_dict = {"x": (1, 32)}
mod = from_onnx(onnx_model, shape_dict=shape_dict)
mod = tvm.relax.transform.LegalizeOps()(mod)

target = tvm.target.Target({"kind": "llvm", "mcpu": "native"})

with tvm.transform.PassContext(opt_level=3):
    ex = tvm.relax.build(mod, target=target)

ex.export_library("/work/compiler/layer2_only_compiled.so")
print("Compiled layer2_only_compiled.so")
# CFD Surrogate on ARM

Deploying a CFD surrogate model to AArch64 with a hand-tuned NEON kernel,
benchmarked against generic compiler codegen.

## Structure
- `model/`     — training + ONNX export of the surrogate model
- `compiler/`  — TVM/ExecuTorch compilation pipeline
- `kernels/`   — hand-written NEON kernel(s) and correctness tests
- `bench/`     — benchmarking scripts and results
- `ci/`        — CI configuration

## Status
Phase 0 complete: cross-compilation toolchain working, verified via Docker
(linux/arm64 container, native on Apple Silicon).

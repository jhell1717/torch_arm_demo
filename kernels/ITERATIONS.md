Iteration 1: naive NEON (vectorized inner loop, vfmaq_f32, no unrolling/tiling)one 
Scalar: 204.30 ns/call
NEON:   29.19 ns/call
Speedup: 7.00x
(sink = 0.000000, ignore this value, it just prevents dead-code elimination)

Iteration 2: unrolled NEON (2 output neurons per iteration, shared x load)
  NEON unrolled: 31.12 ns/call
  Speedup vs scalar: 6.31x
  Speedup vs naive NEON: 0.96x (marginal regression)
  Correctness: max diff ~6e-8 vs PyTorch reference

  Conclusion: for an op this small (32x32, fits entirely in L1 cache),
  naive NEON already saturates the achievable gain on Apple Silicon's
  wide out-of-order core. Manual unrolling did not help further, likely
  because the CPU was already extracting available instruction-level
  parallelism without needing the source-level restructuring. Kept the
  naive NEON version as the final kernel for this op.
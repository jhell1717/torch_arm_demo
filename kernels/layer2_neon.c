#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#define IN_DIM 32
#define OUT_DIM 32

void read_bin(const char *path, float *buf, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); exit(1); }
    size_t read = fread(buf, sizeof(float), n, f);
    if (read != (size_t)n) { fprintf(stderr, "Short read on %s\n", path); exit(1); }
    fclose(f);
}

// Scalar version kept as the correctness reference within this file too
void linear_relu_scalar(const float *W, const float *b, const float *x, float *out) {
    for (int o = 0; o < OUT_DIM; o++) {
        float acc = b[o];
        for (int i = 0; i < IN_DIM; i++) {
            acc += W[o * IN_DIM + i] * x[i];
        }
        out[o] = acc > 0.0f ? acc : 0.0f;
    }
}

// NEON version: vectorize the inner dot-product loop, 4 floats at a time
void linear_relu_neon(const float *W, const float *b, const float *x, float *out) {
    for (int o = 0; o < OUT_DIM; o++) {
        float32x4_t acc_vec = vdupq_n_f32(0.0f);
        const float *w_row = W + o * IN_DIM;

        for (int i = 0; i < IN_DIM; i += 4) {
            float32x4_t w_vec = vld1q_f32(w_row + i);
            float32x4_t x_vec = vld1q_f32(x + i);
            acc_vec = vfmaq_f32(acc_vec, w_vec, x_vec);  // acc += w * x, fused multiply-add
        }

        // Horizontal sum of the 4 lanes in acc_vec
        float32x2_t sum_pair = vadd_f32(vget_low_f32(acc_vec), vget_high_f32(acc_vec));
        sum_pair = vpadd_f32(sum_pair, sum_pair);
        float acc = vget_lane_f32(sum_pair, 0) + b[o];

        out[o] = acc > 0.0f ? acc : 0.0f;
    }
}

// Unrolled version: process 2 output neurons per iteration to hide FMA latency
void linear_relu_neon_unrolled(const float *W, const float *b, const float *x, float *out) {
    for (int o = 0; o < OUT_DIM; o += 2) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        const float *w_row0 = W + o * IN_DIM;
        const float *w_row1 = W + (o + 1) * IN_DIM;

        for (int i = 0; i < IN_DIM; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);  // loaded once, used by both accumulators
            float32x4_t w0_vec = vld1q_f32(w_row0 + i);
            float32x4_t w1_vec = vld1q_f32(w_row1 + i);
            acc0 = vfmaq_f32(acc0, w0_vec, x_vec);
            acc1 = vfmaq_f32(acc1, w1_vec, x_vec);
        }

        float32x2_t sum0 = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
        sum0 = vpadd_f32(sum0, sum0);
        float result0 = vget_lane_f32(sum0, 0) + b[o];

        float32x2_t sum1 = vadd_f32(vget_low_f32(acc1), vget_high_f32(acc1));
        sum1 = vpadd_f32(sum1, sum1);
        float result1 = vget_lane_f32(sum1, 0) + b[o + 1];

        out[o]     = result0 > 0.0f ? result0 : 0.0f;
        out[o + 1] = result1 > 0.0f ? result1 : 0.0f;
    }
}


int main() {
    float W[OUT_DIM * IN_DIM];
    float b[OUT_DIM];
    float x[IN_DIM];
    float out_scalar[OUT_DIM];
    float out_neon[OUT_DIM];
    float ref[OUT_DIM];

    read_bin("layer2_weight.bin", W, OUT_DIM * IN_DIM);
    read_bin("layer2_bias.bin", b, OUT_DIM);
    read_bin("layer2_test_input.bin", x, IN_DIM);
    read_bin("layer2_ref_output.bin", ref, OUT_DIM);

    // --- Correctness check: NEON vs PyTorch reference ---
    linear_relu_neon(W, b, x, out_neon);

    float max_diff = 0.0f;
    for (int o = 0; o < OUT_DIM; o++) {
        float diff = fabsf(out_neon[o] - ref[o]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("NEON vs PyTorch reference max diff: %.8f\n", max_diff);
    if (max_diff < 1e-4f) {
        printf("PASSED: NEON implementation matches PyTorch reference.\n\n");
    } else {
        printf("FAILED: diff exceeds tolerance.\n");
        return 1;
    }

    // --- Correctness check: unrolled NEON vs PyTorch reference ---
    float out_neon_unrolled[OUT_DIM];
    linear_relu_neon_unrolled(W, b, x, out_neon_unrolled);

    float max_diff_unrolled = 0.0f;
    for (int o = 0; o < OUT_DIM; o++) {
        float diff = fabsf(out_neon_unrolled[o] - ref[o]);
        if (diff > max_diff_unrolled) max_diff_unrolled = diff;
    }
    printf("Unrolled NEON vs PyTorch reference max diff: %.8f\n", max_diff_unrolled);
    if (max_diff_unrolled < 1e-4f) {
        printf("PASSED: unrolled NEON implementation matches PyTorch reference.\n\n");
    } else {
        printf("FAILED: diff exceeds tolerance.\n");
        return 1;
    }

// --- Benchmark: scalar vs NEON, same conditions ---
    int n_iters = 1000000;
    struct timespec start, end;
    volatile float sink = 0.0f;  // prevents the optimizer from deleting the loop

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < n_iters; iter++) {
        linear_relu_scalar(W, b, x, out_scalar);
        sink += out_scalar[0];  // forces the result to be "used"
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double scalar_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double scalar_ns = (scalar_sec / n_iters) * 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < n_iters; iter++) {
        linear_relu_neon(W, b, x, out_neon);
        sink += out_neon[0];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double neon_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double neon_ns = (neon_sec / n_iters) * 1e9;

    printf("Scalar: %.2f ns/call\n", scalar_ns);
    printf("NEON:   %.2f ns/call\n", neon_ns);
    printf("Speedup: %.2fx\n", scalar_ns / neon_ns);
    printf("(sink = %f, ignore this value, it just prevents dead-code elimination)\n", sink);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < n_iters; iter++) {
        linear_relu_neon_unrolled(W, b, x, out_neon_unrolled);
        sink += out_neon_unrolled[0];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double unrolled_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double unrolled_ns = (unrolled_sec / n_iters) * 1e9;

    printf("Unrolled NEON: %.2f ns/call\n", unrolled_ns);
    printf("Speedup vs scalar:  %.2fx\n", scalar_ns / unrolled_ns);
    printf("Speedup vs naive NEON: %.2fx\n", neon_ns / unrolled_ns);

    return 0;
}
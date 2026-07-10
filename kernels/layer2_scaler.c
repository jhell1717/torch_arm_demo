#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IN_DIM 32
#define OUT_DIM 32

// Reads a raw float32 binary file into a pre-allocated buffer.
void read_bin(const char *path, float *buf, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); exit(1); }
    size_t read = fread(buf, sizeof(float), n, f);
    if (read != (size_t)n) { fprintf(stderr, "Short read on %s\n", path); exit(1); }
    fclose(f);
}

// out = relu(W @ x + b)
// W is (OUT_DIM, IN_DIM) row-major -- matches PyTorch's nn.Linear.weight layout
void linear_relu_scalar(const float *W, const float *b, const float *x, float *out) {
    for (int o = 0; o < OUT_DIM; o++) {
        float acc = b[o];
        for (int i = 0; i < IN_DIM; i++) {
            acc += W[o * IN_DIM + i] * x[i];
        }
        out[o] = acc > 0.0f ? acc : 0.0f;  // ReLU
    }
}

int main() {
    float W[OUT_DIM * IN_DIM];
    float b[OUT_DIM];
    float x[IN_DIM];
    float out[OUT_DIM];
    float ref[OUT_DIM];

    read_bin("layer2_weight.bin", W, OUT_DIM * IN_DIM);
    read_bin("layer2_bias.bin", b, OUT_DIM);
    read_bin("layer2_test_input.bin", x, IN_DIM);
    read_bin("layer2_ref_output.bin", ref, OUT_DIM);

    linear_relu_scalar(W, b, x, out);

    float max_diff = 0.0f;
    for (int o = 0; o < OUT_DIM; o++) {
        float diff = fabsf(out[o] - ref[o]);
        if (diff > max_diff) max_diff = diff;
        printf("out[%2d] = %.6f   ref[%2d] = %.6f   diff = %.8f\n", o, out[o], o, ref[o], diff);
    }

    printf("\nMax diff: %.8f\n", max_diff);
    if (max_diff < 1e-4f) {
        printf("PASSED: scalar C implementation matches PyTorch reference.\n");
    } else {
        printf("FAILED: diff exceeds tolerance.\n");
        return 1;
    }
    // --- Benchmark ---
    int n_iters = 1000000;
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < n_iters; iter++) {
        linear_relu_scalar(W, b, x, out);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double ns_per_call = (elapsed_sec / n_iters) * 1e9;

    printf("\nBenchmark: %d iterations in %.4f sec\n", n_iters, elapsed_sec);
    printf("Average time per call: %.2f ns\n", ns_per_call);

}
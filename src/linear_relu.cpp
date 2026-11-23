#include <algorithm>
#include <stdexcept>

#include "matmul/matmul.hpp"

namespace matmul {

namespace {

// x < 0 ? 0 : x rather than x > 0 ? x : 0: a NaN fails the comparison and passes through.
inline float relu1(float x) { return x < 0.0f ? 0.0f : x; }

void check_threads(int threads) {
    if (threads < 1) {
        throw std::invalid_argument("matmul: threads must be >= 1");
    }
}

void check_tile_and_threads(int tile, int threads) {
    if (tile < 1) {
        throw std::invalid_argument("matmul: tile must be >= 1");
    }
    check_threads(threads);
}

// One C tile, computed exactly like in tiled_omp. If bias is not null (fused variant), the
// tile also gets bias and ReLU once its last k block is done: at that point the tile
// (tile x tile floats, 64 KiB for T = 128) has just been written and is still in this
// core's L2.
void compute_c_tile(const float* A, const float* B, const float* bias, float* C, int n, int tile,
                    int ii, int jj) {
    const int i_end = std::min(ii + tile, n);
    const int j_end = std::min(jj + tile, n);
    for (int i = ii; i < i_end; ++i) {
        std::fill(C + i * n + jj, C + i * n + j_end, 0.0f);
    }
    for (int kk = 0; kk < n; kk += tile) {
        const int k_end = std::min(kk + tile, n);
        for (int i = ii; i < i_end; ++i) {
            float* Ci = C + i * n;
            for (int k = kk; k < k_end; ++k) {
                const float a = A[i * n + k];
                const float* Bk = B + k * n;
                for (int j = jj; j < j_end; ++j) {
                    Ci[j] += a * Bk[j];
                }
            }
        }
    }
    if (bias != nullptr) {
        for (int i = ii; i < i_end; ++i) {
            float* Ci = C + i * n;
            for (int j = jj; j < j_end; ++j) {
                Ci[j] = relu1(Ci[j] + bias[j]);
            }
        }
    }
}

// Both variants run this one function, i.e. the same machine code for the matmul part, and
// differ only in a pointer checked once per tile. The expected fusion effect is ~1%, while
// two compiled copies of identical loops can run several percent apart: the first version
// of this stage (a template instantiated twice) did, by 3-8% at T = 64; see docs/NOTES.md.
void matmul_tiles(const float* A, const float* B, const float* bias, float* C, int n, int tile,
                  int threads) {
    const int tiles = (n + tile - 1) / tile;
#pragma omp parallel for collapse(2) schedule(dynamic) num_threads(threads)
    for (int ti = 0; ti < tiles; ++ti) {
        for (int tj = 0; tj < tiles; ++tj) {
            compute_c_tile(A, B, bias, C, n, tile, ti * tile, tj * tile);
        }
    }
}

}  // namespace

void add_bias(float* C, const float* bias, int n, int threads) {
    check_threads(threads);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < n; ++i) {
        float* Ci = C + i * n;
        for (int j = 0; j < n; ++j) {
            Ci[j] += bias[j];
        }
    }
}

void relu(float* C, int n, int threads) {
    check_threads(threads);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < n; ++i) {
        float* Ci = C + i * n;
        for (int j = 0; j < n; ++j) {
            Ci[j] = relu1(Ci[j]);
        }
    }
}

void add_bias_relu(float* C, const float* bias, int n, int threads) {
    check_threads(threads);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < n; ++i) {
        float* Ci = C + i * n;
        for (int j = 0; j < n; ++j) {
            Ci[j] = relu1(Ci[j] + bias[j]);
        }
    }
}

void linear_relu_unfused(const float* A, const float* B, const float* bias, float* C, int n,
                         int tile, int threads) {
    check_tile_and_threads(tile, threads);
    matmul_tiles(A, B, nullptr, C, n, tile, threads);
    add_bias(C, bias, n, threads);
    relu(C, n, threads);
}

void linear_relu_fused(const float* A, const float* B, const float* bias, float* C, int n,
                       int tile, int threads) {
    check_tile_and_threads(tile, threads);
    matmul_tiles(A, B, bias, C, n, tile, threads);
}

}  // namespace matmul

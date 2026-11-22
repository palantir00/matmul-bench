#include <algorithm>
#include <stdexcept>

#include "matmul/matmul.hpp"

namespace matmul {

namespace {

// Computes one tile of C completely: C(ii, jj) = sum over kk of A(ii, kk) * B(kk, jj),
// with the same i-k-j loops as the single-threaded tiled kernel. Only rows ii..i_end-1
// and columns jj..j_end-1 of C are written, so different tiles never overlap.
void compute_c_tile(const float* A, const float* B, float* C, int n, int tile, int ii, int jj) {
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
}

}  // namespace

void tiled_omp(const float* A, const float* B, float* C, int n, int tile, int threads,
               Schedule schedule) {
    if (tile < 1) {
        throw std::invalid_argument("matmul::tiled_omp: tile must be >= 1");
    }
    if (threads < 1) {
        throw std::invalid_argument("matmul::tiled_omp: threads must be >= 1");
    }
    const int tiles = (n + tile - 1) / tile;  // tiles per dimension, the last may be smaller

    // collapse(2) turns the tiles x tiles grid into one list of tasks, so there is enough
    // work to share even when one dimension has fewer tiles than there are threads.
    // The two branches differ only in the schedule clause (it must be known at compile
    // time, unless schedule(runtime) is used, which would depend on global state).
    if (schedule == Schedule::Static) {
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
        for (int ti = 0; ti < tiles; ++ti) {
            for (int tj = 0; tj < tiles; ++tj) {
                compute_c_tile(A, B, C, n, tile, ti * tile, tj * tile);
            }
        }
    } else {
#pragma omp parallel for collapse(2) schedule(dynamic) num_threads(threads)
        for (int ti = 0; ti < tiles; ++ti) {
            for (int tj = 0; tj < tiles; ++tj) {
                compute_c_tile(A, B, C, n, tile, ti * tile, tj * tile);
            }
        }
    }
}

}  // namespace matmul

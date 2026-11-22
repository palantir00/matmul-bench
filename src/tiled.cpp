#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "matmul/matmul.hpp"

namespace matmul {

void tiled(const float* A, const float* B, float* C, int n, int tile) {
    if (tile < 1) {
        throw std::invalid_argument("matmul::tiled: tile must be >= 1");
    }
    std::fill(C, C + static_cast<std::size_t>(n) * n, 0.0f);

    for (int ii = 0; ii < n; ii += tile) {
        const int i_end = std::min(ii + tile, n);
        for (int jj = 0; jj < n; jj += tile) {
            const int j_end = std::min(jj + tile, n);
            // The C tile (ii, jj) receives all its terms here, so it stays hot in cache
            // for the whole kk loop.
            for (int kk = 0; kk < n; kk += tile) {
                const int k_end = std::min(kk + tile, n);
                // C tile (ii, jj) += A tile (ii, kk) * B tile (kk, jj), in i-k-j order.
                // The B tile is reused by every row i of the tile while it is in cache.
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
    }
    // kk grows in the outer loop and k grows inside a tile, so every C[i][j] still gets
    // its terms in order k = 0, 1, ..., n-1: the same summation order as ikj.
}

}  // namespace matmul

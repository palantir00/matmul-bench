#include "matmul/matmul.hpp"

namespace matmul {

void ikj(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        float* Ci = C + i * n;  // row i of C: the accumulator, reused for every k
        for (int j = 0; j < n; ++j) {
            Ci[j] = 0.0f;
        }
        for (int k = 0; k < n; ++k) {
            const float a = A[i * n + k];   // one scalar of A, broadcast over the row
            const float* Bk = B + k * n;    // row k of B, read front to back
            // C[i][j] still receives its terms in order k = 0, 1, ..., n-1, exactly like
            // the naive dot product. The j iterations are independent, so the compiler
            // may run 8 of them at once in one AVX2 register without changing any result.
            for (int j = 0; j < n; ++j) {
                Ci[j] += a * Bk[j];
            }
        }
    }
}

}  // namespace matmul

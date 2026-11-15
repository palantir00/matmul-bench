#include "matmul/matmul.hpp"

namespace matmul {

void naive(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // One dot product: row i of A with column j of B.
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                // A[i][k] walks consecutive addresses; B[k][j] jumps n floats per step,
                // so every B access lands on a different cache line.
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

}  // namespace matmul

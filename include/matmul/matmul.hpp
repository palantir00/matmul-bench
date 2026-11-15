#pragma once

// Matrix multiplication kernels, one per optimization step.
//
// Conventions shared by every kernel:
//   - A, B and C are n x n, row-major, float32: element (i, j) lives at index i * n + j
//   - C = A * B; C is overwritten and its previous contents are ignored
//   - A, B and C must not overlap
namespace matmul {

// Version 1: textbook i-j-k loop order. B is walked down a column (stride n floats).
void naive(const float* A, const float* B, float* C, int n);

}  // namespace matmul

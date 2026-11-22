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

// Version 2: i-k-j loop order. The inner loop adds a * (row k of B) to row i of C,
// so every access is contiguous and the loop vectorizes without reordering the sum.
void ikj(const float* A, const float* B, float* C, int n);

// Version 3: blocked (tiled) i-k-j. C is computed tile by tile from the block formula
// C_IJ = sum_K A_IK * B_KJ with tile x tile blocks; each block product runs in i-k-j order.
// Edge tiles are smaller when tile does not divide n. Throws std::invalid_argument if tile < 1.
void tiled(const float* A, const float* B, float* C, int n, int tile);

}  // namespace matmul

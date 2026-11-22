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

// How OpenMP hands out the C tiles to threads.
enum class Schedule {
    Static,   // each thread gets an equal, fixed share of the tiles up front
    Dynamic,  // a thread takes the next tile whenever it finishes one
};

// Version 4: tiled, multithreaded with OpenMP. Every C tile is one task computed by one
// thread from start to end, so threads never write to the same memory and every element
// gets its terms in the same order as in ikj (results are identical for any thread count).
// Throws std::invalid_argument if tile < 1 or threads < 1.
void tiled_omp(const float* A, const float* B, float* C, int n, int tile, int threads,
               Schedule schedule = Schedule::Dynamic);

}  // namespace matmul

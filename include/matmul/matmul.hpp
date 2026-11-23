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

// Version 5: linear layer with ReLU, C = relu(A * B + bias), like nn.Linear + ReLU with
// A = inputs and B = weights. bias has n elements; bias[j] is added to every element of
// column j. The matmul part is the tiled OpenMP kernel (dynamic schedule).
//
// Unfused: matmul, then a bias pass and a ReLU pass over C (three passes over C, like
// three separate kernels in an eager framework).
// Fused: bias and ReLU are applied to each C tile right after its last k block, while
// the tile is still in cache (one pass over C).
// Both do the same operations on every element in the same order, so their results are
// identical bit for bit. Throw std::invalid_argument if tile < 1 or threads < 1.
void linear_relu_unfused(const float* A, const float* B, const float* bias, float* C, int n,
                         int tile, int threads);
void linear_relu_fused(const float* A, const float* B, const float* bias, float* C, int n,
                       int tile, int threads);

// Elementwise passes over an n x n matrix, in place, rows split over threads.
// ReLU keeps NaN as NaN, so an error upstream is not hidden as 0.
// Throw std::invalid_argument if threads < 1.
void add_bias(float* C, const float* bias, int n, int threads);       // C(i, j) += bias[j]
void relu(float* C, int n, int threads);                              // C = max(C, 0)
void add_bias_relu(float* C, const float* bias, int n, int threads);  // both in one pass

}  // namespace matmul

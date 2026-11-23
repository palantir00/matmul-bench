#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "matmul/matmul.hpp"
#include "matmul/matrix.hpp"

namespace {

using matmul::Matrix;
using Kernel = void (*)(const float*, const float*, float*, int);

struct KernelCase {
    const char* name;
    Kernel fn;
};

// Lets GoogleTest print the parameter by name in failure messages.
void PrintTo(const KernelCase& k, std::ostream* os) { *os << k.name; }

// Every kernel version, naive first. A new version needs one line here and is then
// covered by all parametrized tests below.
// Kernels with an extra parameter are wrapped in captureless lambdas, which convert to a
// plain function pointer. Tile sizes 1 and 3 divide almost none of the test sizes, so
// they exercise the smaller edge tiles; 64 is larger than the small matrices.
const KernelCase kAllKernels[] = {
    {"naive", matmul::naive},
    {"ikj", matmul::ikj},
    {"tiled_t1", [](const float* A, const float* B, float* C, int n) { matmul::tiled(A, B, C, n, 1); }},
    {"tiled_t3", [](const float* A, const float* B, float* C, int n) { matmul::tiled(A, B, C, n, 3); }},
    {"tiled_t16", [](const float* A, const float* B, float* C, int n) { matmul::tiled(A, B, C, n, 16); }},
    {"tiled_t64", [](const float* A, const float* B, float* C, int n) { matmul::tiled(A, B, C, n, 64); }},
    // More threads than tiles for small n, and an odd thread count that does not divide
    // the number of tiles.
    {"omp_t16_x4", [](const float* A, const float* B, float* C, int n) {
         matmul::tiled_omp(A, B, C, n, 16, 4);
     }},
    {"omp_t3_x7_static", [](const float* A, const float* B, float* C, int n) {
         matmul::tiled_omp(A, B, C, n, 3, 7, matmul::Schedule::Static);
     }},
};

// Different seeds for A and B: with A == B, a kernel that computes B * A instead of A * B
// (or mixes up i and j) would still pass.
constexpr unsigned kSeedA = 42;
constexpr unsigned kSeedB = 43;

// Reference value of every element of A * B plus its error scale
// scale_ij = sum_k |a_ik * b_kj|. The rounding error of a float dot product is bounded
// relative to this sum, not to the result: when terms cancel, the result can be ~0
// while the accumulated rounding error is not.
struct Expected {
    std::vector<double> value;
    std::vector<double> scale;
};

// Reference computed in double precision.
Expected exact_product(const Matrix& A, const Matrix& B) {
    const int n = A.n();
    Expected e{std::vector<double>(A.size()), std::vector<double>(A.size())};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            double abs_sum = 0.0;
            for (int k = 0; k < n; ++k) {
                const double p = static_cast<double>(A(i, k)) * static_cast<double>(B(k, j));
                sum += p;
                abs_sum += std::abs(p);
            }
            const std::size_t idx = static_cast<std::size_t>(i) * n + j;
            e.value[idx] = sum;
            e.scale[idx] = abs_sum;
        }
    }
    return e;
}

// Reference taken from another float result R (the naive kernel), with the exact scale.
Expected as_reference(const Matrix& R, const Expected& exact) {
    return Expected{std::vector<double>(R.data(), R.data() + R.size()), exact.scale};
}

// Worst-case rounding error of a length-n float dot product is about n * u * scale,
// with unit roundoff u = 2^-24. We allow n * eps = 2 * n * u (eps = 2^-23).
double tolerance(int n) {
    return n * static_cast<double>(std::numeric_limits<float>::epsilon());
}

// Largest |C_ij - reference_ij| / scale_ij over the matrix (how close we get to the bound).
double max_error_ratio(const Matrix& C, const Expected& e) {
    double worst = 0.0;
    for (std::size_t idx = 0; idx < C.size(); ++idx) {
        if (e.scale[idx] > 0.0) {
            const double err = std::abs(static_cast<double>(C.data()[idx]) - e.value[idx]);
            worst = std::max(worst, err / e.scale[idx]);
        }
    }
    return worst;
}

// Checks |C_ij - reference_ij| <= rtol * scale_ij for every element.
::testing::AssertionResult close_to(const Matrix& C, const Expected& e, double rtol) {
    const int n = C.n();
    int failures = 0;
    std::ostringstream first;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const std::size_t idx = static_cast<std::size_t>(i) * n + j;
            const double err = std::abs(static_cast<double>(C(i, j)) - e.value[idx]);
            const double bound = rtol * e.scale[idx];
            if (!(err <= bound)) {  // written this way so that NaN also fails
                if (failures < 5) {
                    first << "\n  C(" << i << "," << j << ") = " << C(i, j) << ", expected "
                          << e.value[idx] << ", |error| " << err << " > bound " << bound;
                }
                ++failures;
            }
        }
    }
    if (failures == 0) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << failures << " of " << C.size() << " elements out of tolerance:" << first.str();
}

std::string kernel_name(const ::testing::TestParamInfo<KernelCase>& info) {
    return info.param.name;
}

std::string kernel_size_name(const ::testing::TestParamInfo<std::tuple<KernelCase, int>>& info) {
    return std::string(std::get<0>(info.param).name) + "_n" +
           std::to_string(std::get<1>(info.param));
}

// Sizes: tiny, odd and powers of two. Odd sizes matter for blocked versions, where the
// last block is smaller than the others.
const int kSizes[] = {1, 2, 7, 16, 64, 127, 129, 256};

// ---- Basic properties, run for every kernel -----------------------------------

class KernelBasics : public ::testing::TestWithParam<KernelCase> {};

TEST_P(KernelBasics, HandComputed2x2) {
    // [1 2]   [5 6]   [19 22]
    // [3 4] x [7 8] = [43 50]      (small integers are exact in float)
    Matrix A(2), B(2), C(2);
    const float a[] = {1, 2, 3, 4};
    const float b[] = {5, 6, 7, 8};
    std::copy(a, a + 4, A.data());
    std::copy(b, b + 4, B.data());

    GetParam().fn(A.data(), B.data(), C.data(), 2);

    EXPECT_EQ(C(0, 0), 19.0f);
    EXPECT_EQ(C(0, 1), 22.0f);
    EXPECT_EQ(C(1, 0), 43.0f);
    EXPECT_EQ(C(1, 1), 50.0f);  // B * A would give [23 34; 31 46]
}

TEST_P(KernelBasics, IdentityIsExact) {
    // Multiplying by 1 and adding 0 never rounds, so A * I and I * A equal A bit for bit.
    const int n = 33;
    Matrix A(n), I(n), C(n);
    A.fill_random(kSeedA);
    I.fill(0.0f);
    for (int i = 0; i < n; ++i) {
        I(i, i) = 1.0f;
    }

    GetParam().fn(A.data(), I.data(), C.data(), n);
    for (std::size_t idx = 0; idx < A.size(); ++idx) {
        ASSERT_EQ(C.data()[idx], A.data()[idx]) << "A * I differs at index " << idx;
    }
    GetParam().fn(I.data(), A.data(), C.data(), n);
    for (std::size_t idx = 0; idx < A.size(); ++idx) {
        ASSERT_EQ(C.data()[idx], A.data()[idx]) << "I * A differs at index " << idx;
    }
}

TEST_P(KernelBasics, OverwritesC) {
    // Old contents of C must be ignored. NaN poisons any kernel that accumulates into C
    // without zeroing it first (i-k-j and later versions accumulate into C).
    const int n = 17;
    Matrix A(n), B(n), C(n);
    A.fill_random(kSeedA);
    B.fill_random(kSeedB);
    C.fill(std::numeric_limits<float>::quiet_NaN());

    GetParam().fn(A.data(), B.data(), C.data(), n);

    EXPECT_TRUE(close_to(C, exact_product(A, B), tolerance(n)));
}

INSTANTIATE_TEST_SUITE_P(All, KernelBasics, ::testing::ValuesIn(kAllKernels), kernel_name);

// ---- Accuracy against the double-precision reference --------------------------

class KernelVsExact : public ::testing::TestWithParam<std::tuple<KernelCase, int>> {};

TEST_P(KernelVsExact, WithinRoundingBound) {
    const auto [kernel, n] = GetParam();
    Matrix A(n), B(n), C(n);
    A.fill_random(kSeedA);
    B.fill_random(kSeedB);

    kernel.fn(A.data(), B.data(), C.data(), n);

    const Expected e = exact_product(A, B);
    EXPECT_TRUE(close_to(C, e, tolerance(n)));
    std::printf("[ info     ] %-6s n=%4d  max |error|/scale = %.2e   (bound n*eps = %.2e)\n",
                kernel.name, n, max_error_ratio(C, e), tolerance(n));
}

INSTANTIATE_TEST_SUITE_P(Sizes, KernelVsExact,
                         ::testing::Combine(::testing::ValuesIn(kAllKernels),
                                            ::testing::ValuesIn(kSizes)),
                         kernel_size_name);

// ---- Every optimized version against naive ------------------------------------
// Naive is the reference that is obviously correct by reading it (and is itself checked
// against double above). Each optimization must agree with it up to rounding.

class KernelVsNaive : public ::testing::TestWithParam<std::tuple<KernelCase, int>> {};

TEST_P(KernelVsNaive, WithinTwiceTheBound) {
    const auto [kernel, n] = GetParam();
    Matrix A(n), B(n), C(n), R(n);
    A.fill_random(kSeedA);
    B.fill_random(kSeedB);

    kernel.fn(A.data(), B.data(), C.data(), n);
    matmul::naive(A.data(), B.data(), R.data(), n);

    // Both results lie within n*eps*scale of the exact value, so they lie within
    // 2*n*eps*scale of each other (triangle inequality).
    const Expected ref = as_reference(R, exact_product(A, B));
    EXPECT_TRUE(close_to(C, ref, 2 * tolerance(n)));

    std::size_t identical = 0;
    for (std::size_t idx = 0; idx < C.size(); ++idx) {
        identical += (C.data()[idx] == R.data()[idx]) ? 1 : 0;
    }
    std::printf("[ info     ] %-6s n=%4d  bit-identical to naive: %5.1f%%   max |diff|/scale = %.2e\n",
                kernel.name, n, 100.0 * identical / C.size(), max_error_ratio(C, ref));
}

INSTANTIATE_TEST_SUITE_P(Sizes, KernelVsNaive,
                         ::testing::Combine(  // skip index 0: naive against itself
                             ::testing::ValuesIn(std::next(std::begin(kAllKernels)),
                                                 std::end(kAllKernels)),
                             ::testing::ValuesIn(kSizes)),
                         kernel_size_name);

// ---- Tiled-specific -----------------------------------------------------------

// Tiling in the ii-jj-kk order does not reorder the sum: every C[i][j] still gets its
// terms in order k = 0..n-1, each as one fused multiply-add, exactly as in ikj. So the
// results must be identical bit for bit, for any tile size. (This relies on both kernels
// being built with the same flags, so that both use FMA or both do not.)
TEST(Tiled, BitIdenticalToIkj) {
    for (int n : {127, 256}) {
        Matrix A(n), B(n), R(n), C(n);
        A.fill_random(kSeedA);
        B.fill_random(kSeedB);
        matmul::ikj(A.data(), B.data(), R.data(), n);
        for (int tile : {1, 3, 16, 64, 100, 256}) {
            matmul::tiled(A.data(), B.data(), C.data(), n, tile);
            std::size_t mismatches = 0;
            for (std::size_t idx = 0; idx < C.size(); ++idx) {
                mismatches += (C.data()[idx] != R.data()[idx]) ? 1 : 0;
            }
            EXPECT_EQ(mismatches, 0u) << "n=" << n << " tile=" << tile;
        }
    }
}

TEST(Tiled, RejectsNonPositiveTile) {
    Matrix A(4), B(4), C(4);
    EXPECT_THROW(matmul::tiled(A.data(), B.data(), C.data(), 4, 0), std::invalid_argument);
    EXPECT_THROW(matmul::tiled(A.data(), B.data(), C.data(), 4, -8), std::invalid_argument);
}

// ---- OpenMP-specific -----------------------------------------------------------

// Each C tile is computed by one thread in the same order as ikj, so the thread count and
// the schedule must not change a single bit. A race (two threads writing one tile) or a
// skipped tile would show up here as mismatches.
TEST(TiledOmp, BitIdenticalToIkjForAnyThreadCount) {
    const int n = 129;  // 129 = 2 * 64 + 1: a one-element-wide edge tile
    Matrix A(n), B(n), R(n), C(n);
    A.fill_random(kSeedA);
    B.fill_random(kSeedB);
    matmul::ikj(A.data(), B.data(), R.data(), n);

    for (auto schedule : {matmul::Schedule::Static, matmul::Schedule::Dynamic}) {
        for (int tile : {16, 64}) {
            for (int threads : {1, 2, 3, 7, 16}) {
                C.fill(std::numeric_limits<float>::quiet_NaN());
                matmul::tiled_omp(A.data(), B.data(), C.data(), n, tile, threads, schedule);
                std::size_t mismatches = 0;
                for (std::size_t idx = 0; idx < C.size(); ++idx) {
                    mismatches += (C.data()[idx] != R.data()[idx]) ? 1 : 0;
                }
                EXPECT_EQ(mismatches, 0u)
                    << "tile=" << tile << " threads=" << threads << " static="
                    << (schedule == matmul::Schedule::Static);
            }
        }
    }
}

TEST(TiledOmp, RejectsBadArguments) {
    Matrix A(4), B(4), C(4);
    EXPECT_THROW(matmul::tiled_omp(A.data(), B.data(), C.data(), 4, 0, 2), std::invalid_argument);
    EXPECT_THROW(matmul::tiled_omp(A.data(), B.data(), C.data(), 4, 2, 0), std::invalid_argument);
}

// ---- Linear layer with ReLU (stage 5) -------------------------------------------

constexpr unsigned kSeedBias = 44;

std::vector<float> random_bias(int n) {
    std::mt19937 gen(kSeedBias);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> bias(n);
    for (float& b : bias) {
        b = dist(gen);
    }
    return bias;
}

// Reference for relu(A * B + bias) in double. Adding the bias is one more rounded addition,
// like an (n+1)-th term of the dot product: the scale gets |bias_j| and the bound for n + 1
// terms applies. ReLU never increases a difference (|relu(x) - relu(y)| <= |x - y|), so it
// needs no extra margin.
Expected exact_linear_relu(const Matrix& A, const Matrix& B, const std::vector<float>& bias) {
    Expected e = exact_product(A, B);
    const int n = A.n();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const std::size_t idx = static_cast<std::size_t>(i) * n + j;
            e.value[idx] = std::max(e.value[idx] + static_cast<double>(bias[j]), 0.0);
            e.scale[idx] += std::abs(static_cast<double>(bias[j]));
        }
    }
    return e;
}

using LinearKernel = void (*)(const float*, const float*, const float*, float*, int, int, int);

struct LinearCase {
    const char* name;
    LinearKernel fn;
};

const LinearCase kLinearKernels[] = {
    {"unfused", matmul::linear_relu_unfused},
    {"fused", matmul::linear_relu_fused},
};

// (tile, threads): edge tiles and more threads than tiles, as for tiled_omp above.
const std::pair<int, int> kTileThreads[] = {{3, 7}, {16, 4}, {64, 1}};

std::size_t count_mismatches(const Matrix& X, const Matrix& Y) {
    std::size_t mismatches = 0;
    for (std::size_t idx = 0; idx < X.size(); ++idx) {
        mismatches += (X.data()[idx] != Y.data()[idx]) ? 1 : 0;
    }
    return mismatches;
}

TEST(LinearRelu, WithinBoundOfExactAndNaive) {
    for (int n : kSizes) {
        Matrix A(n), B(n), C(n), R(n);
        A.fill_random(kSeedA);
        B.fill_random(kSeedB);
        const std::vector<float> bias = random_bias(n);
        const Expected exact = exact_linear_relu(A, B, bias);

        // Naive reference: the textbook matmul, then bias and ReLU written out here.
        matmul::naive(A.data(), B.data(), R.data(), n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                R(i, j) = std::max(R(i, j) + bias[j], 0.0f);
            }
        }
        const Expected ref = as_reference(R, exact);

        for (const LinearCase& k : kLinearKernels) {
            for (const auto& [tile, threads] : kTileThreads) {
                C.fill(std::numeric_limits<float>::quiet_NaN());
                k.fn(A.data(), B.data(), bias.data(), C.data(), n, tile, threads);
                EXPECT_TRUE(close_to(C, exact, tolerance(n + 1)))
                    << k.name << " n=" << n << " tile=" << tile << " threads=" << threads;
                EXPECT_TRUE(close_to(C, ref, 2 * tolerance(n + 1)))
                    << k.name << " vs naive, n=" << n << " tile=" << tile;
            }
        }
    }
}

// Fused and unfused do the same float operations on every element in the same order:
// the dot product as in ikj, then + bias[j], then ReLU. Only the moment of the epilogue
// differs, so the results must agree bit for bit for any tile size and thread count.
TEST(LinearRelu, FusedBitIdenticalToUnfused) {
    for (int n : {127, 256}) {
        Matrix A(n), B(n), R(n), C(n);
        A.fill_random(kSeedA);
        B.fill_random(kSeedB);
        const std::vector<float> bias = random_bias(n);
        for (int tile : {1, 16, 64, 100}) {
            for (int threads : {1, 3, 16}) {
                R.fill(std::numeric_limits<float>::quiet_NaN());
                C.fill(std::numeric_limits<float>::quiet_NaN());
                matmul::linear_relu_unfused(A.data(), B.data(), bias.data(), R.data(), n, tile,
                                            threads);
                matmul::linear_relu_fused(A.data(), B.data(), bias.data(), C.data(), n, tile,
                                          threads);
                EXPECT_EQ(count_mismatches(C, R), 0u)
                    << "n=" << n << " tile=" << tile << " threads=" << threads;
            }
        }
    }
}

TEST(LinearRelu, BiasGoesToColumnsAndReluClamps) {
    // I * B = B, then bias per column: [1 2; 3 4] + [10 -3] = [11 -1; 13 1] -> ReLU.
    // A bias added per row instead would give [11 12; 0 1].
    Matrix I(2), B(2), C(2);
    const float b[] = {1, 2, 3, 4};
    std::copy(b, b + 4, B.data());
    I.fill(0.0f);
    I(0, 0) = I(1, 1) = 1.0f;
    const float bias[] = {10.0f, -3.0f};

    for (const LinearCase& k : kLinearKernels) {
        k.fn(I.data(), B.data(), bias, C.data(), 2, 1, 2);
        EXPECT_EQ(C(0, 0), 11.0f) << k.name;
        EXPECT_EQ(C(0, 1), 0.0f) << k.name;
        EXPECT_EQ(C(1, 0), 13.0f) << k.name;
        EXPECT_EQ(C(1, 1), 1.0f) << k.name;
    }
}

TEST(LinearRelu, RejectsBadArguments) {
    Matrix A(4), B(4), C(4);
    const float bias[4] = {};
    for (const LinearCase& k : kLinearKernels) {
        EXPECT_THROW(k.fn(A.data(), B.data(), bias, C.data(), 4, 0, 2), std::invalid_argument);
        EXPECT_THROW(k.fn(A.data(), B.data(), bias, C.data(), 4, 2, 0), std::invalid_argument);
    }
    EXPECT_THROW(matmul::add_bias(C.data(), bias, 4, 0), std::invalid_argument);
    EXPECT_THROW(matmul::relu(C.data(), 4, 0), std::invalid_argument);
    EXPECT_THROW(matmul::add_bias_relu(C.data(), bias, 4, 0), std::invalid_argument);
}

TEST(Elementwise, TwoPassesEqualOnePass) {
    const int n = 37;
    Matrix X(n), Y(n);
    X.fill_random(kSeedA);
    std::copy(X.data(), X.data() + X.size(), Y.data());
    const std::vector<float> bias = random_bias(n);
    for (int threads : {1, 5}) {
        matmul::add_bias(X.data(), bias.data(), n, threads);
        matmul::relu(X.data(), n, threads);
        matmul::add_bias_relu(Y.data(), bias.data(), n, threads);
        EXPECT_EQ(count_mismatches(X, Y), 0u) << "threads=" << threads;
    }
}

TEST(Elementwise, ReluClampsNegativesAndKeepsNaN) {
    Matrix C(2);
    const float c[] = {-2.0f, 0.0f, 3.0f, std::numeric_limits<float>::quiet_NaN()};
    std::copy(c, c + 4, C.data());
    matmul::relu(C.data(), 2, 1);
    EXPECT_EQ(C(0, 0), 0.0f);
    EXPECT_EQ(C(0, 1), 0.0f);
    EXPECT_EQ(C(1, 0), 3.0f);
    EXPECT_TRUE(std::isnan(C(1, 1))) << "ReLU must not turn NaN into 0";
}

// ---- The check itself ---------------------------------------------------------
// It must be strict enough to catch a real bug, e.g. one forgotten term of a sum
// (a typical off-by-one at a block edge).
TEST(Tolerance, CatchesMissingTerm) {
    const int n = 129;
    Matrix A(n), B(n), C(n);
    A.fill_random(kSeedA);
    B.fill_random(kSeedB);
    A(5, 0) = 1.0f;  // make the term we are going to drop clearly non-negligible
    B(0, 7) = 1.0f;

    matmul::naive(A.data(), B.data(), C.data(), n);
    const Expected e = exact_product(A, B);
    ASSERT_TRUE(close_to(C, e, tolerance(n)));

    C(5, 7) -= 1.0f;  // pretend the kernel skipped k = 0 for this one element
    EXPECT_FALSE(close_to(C, e, tolerance(n)));
}

}  // namespace

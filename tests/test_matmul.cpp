#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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

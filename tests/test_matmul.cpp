#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <vector>

#include "matmul/matmul.hpp"
#include "matmul/matrix.hpp"

namespace {

using matmul::Matrix;

// Different seeds for A and B: with A == B, a kernel that computes B * A instead of A * B
// (or mixes up i and j) would still pass.
constexpr unsigned kSeedA = 42;
constexpr unsigned kSeedB = 43;

// Expected value of every element of A * B, computed in double, plus its error scale
// scale_ij = sum_k |a_ik * b_kj|. The rounding error of a float dot product is bounded
// relative to this sum, not to the result: when terms cancel, the result can be ~0
// while the accumulated rounding error is not.
struct Expected {
    std::vector<double> value;
    std::vector<double> scale;
};

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

// Worst-case rounding error of a length-n float dot product is about n * u * scale,
// with unit roundoff u = 2^-24. We allow n * eps = 2 * n * u (eps = 2^-23).
double tolerance(int n) {
    return n * static_cast<double>(std::numeric_limits<float>::epsilon());
}

// Largest |C_ij - expected_ij| / scale_ij over the matrix (how close we get to the bound).
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

// Checks |C_ij - expected_ij| <= rtol * scale_ij for every element.
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

TEST(Naive, HandComputed2x2) {
    // [1 2]   [5 6]   [19 22]
    // [3 4] x [7 8] = [43 50]      (small integers are exact in float)
    Matrix A(2), B(2), C(2);
    const float a[] = {1, 2, 3, 4};
    const float b[] = {5, 6, 7, 8};
    std::copy(a, a + 4, A.data());
    std::copy(b, b + 4, B.data());

    matmul::naive(A.data(), B.data(), C.data(), 2);

    EXPECT_EQ(C(0, 0), 19.0f);
    EXPECT_EQ(C(0, 1), 22.0f);
    EXPECT_EQ(C(1, 0), 43.0f);
    EXPECT_EQ(C(1, 1), 50.0f);  // B * A would give [23 34; 31 46]
}

TEST(Naive, IdentityIsExact) {
    // Multiplying by 1 and adding 0 never rounds, so A * I and I * A equal A bit for bit.
    const int n = 33;
    Matrix A(n), I(n), C(n);
    A.fill_random(kSeedA);
    I.fill(0.0f);
    for (int i = 0; i < n; ++i) {
        I(i, i) = 1.0f;
    }

    matmul::naive(A.data(), I.data(), C.data(), n);
    for (std::size_t idx = 0; idx < A.size(); ++idx) {
        ASSERT_EQ(C.data()[idx], A.data()[idx]) << "A * I differs at index " << idx;
    }
    matmul::naive(I.data(), A.data(), C.data(), n);
    for (std::size_t idx = 0; idx < A.size(); ++idx) {
        ASSERT_EQ(C.data()[idx], A.data()[idx]) << "I * A differs at index " << idx;
    }
}

TEST(Naive, OverwritesC) {
    // Old contents of C must be ignored. NaN poisons any kernel that accumulates into C.
    const int n = 17;
    Matrix A(n), B(n), C(n);
    A.fill_random(kSeedA);
    B.fill_random(kSeedB);
    C.fill(std::numeric_limits<float>::quiet_NaN());

    matmul::naive(A.data(), B.data(), C.data(), n);

    EXPECT_TRUE(close_to(C, exact_product(A, B), tolerance(n)));
}

// Sizes: tiny, odd and powers of two. Odd sizes matter later for blocked versions,
// where the last block is smaller than the others.
class NaiveVsExact : public ::testing::TestWithParam<int> {};

TEST_P(NaiveVsExact, WithinRoundingBound) {
    const int n = GetParam();
    Matrix A(n), B(n), C(n);
    A.fill_random(kSeedA);
    B.fill_random(kSeedB);

    matmul::naive(A.data(), B.data(), C.data(), n);

    const Expected e = exact_product(A, B);
    EXPECT_TRUE(close_to(C, e, tolerance(n)));
    std::printf("[ info     ] n=%4d  max |error|/scale = %.2e   (bound n*eps = %.2e)\n", n,
                max_error_ratio(C, e), tolerance(n));
}

INSTANTIATE_TEST_SUITE_P(Sizes, NaiveVsExact, ::testing::Values(1, 2, 7, 16, 64, 127, 129, 256));

// The check must be strict enough to catch a real bug, e.g. one forgotten term of a sum
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

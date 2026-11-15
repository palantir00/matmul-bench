#include <gtest/gtest.h>

#include <limits>

// Stage 0 smoke test: verifies that GoogleTest is wired into CMake/CTest.
TEST(Hello, ToolchainWorks) {
    EXPECT_EQ(1 + 1, 2);
}

// Every correctness test later on assumes IEEE 754 single precision floats.
// 24 significant bits (23 stored + 1 implicit) give machine epsilon 2^-23 ~ 1.19e-7,
// which is the scale we will use to reason about test tolerances.
TEST(Hello, FloatIsIeee754Single) {
    static_assert(std::numeric_limits<float>::is_iec559, "float must be IEEE 754");
    EXPECT_EQ(sizeof(float), 4u);
    EXPECT_EQ(std::numeric_limits<float>::digits, 24);
}

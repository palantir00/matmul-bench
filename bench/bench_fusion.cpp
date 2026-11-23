// Stage 5: fusing bias + ReLU into the matmul, and the same epilogue on its own.
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <random>

#include "matmul/matmul.hpp"
#include "matmul/matrix.hpp"

namespace {

constexpr unsigned kSeedA = 42;
constexpr unsigned kSeedB = 43;
constexpr unsigned kSeedBias = 44;

matmul::AlignedFloats random_bias(int n) {
    auto bias = matmul::alloc_aligned(static_cast<std::size_t>(n));
    std::mt19937 gen(kSeedBias);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::generate(bias.get(), bias.get() + n, [&] { return dist(gen); });
    return bias;
}

// relu(A * B + bias), fused or unfused. Same counters as the matmul benchmarks: FLOPS counts
// only the 2n^3 of the matmul, so both variants report the same work and GFLOP/s compare
// directly (the epilogue adds 2n^2, i.e. 0.1% at n = 1000).
void BM_LinearRelu(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int tile = static_cast<int>(state.range(1));
    const int threads = static_cast<int>(state.range(2));
    const bool fused = state.range(3) != 0;

    matmul::Matrix A(n), B(n), C(n);
    A.fill_random(kSeedA);
    B.fill_random(kSeedB);
    C.fill(0.0f);
    const auto bias = random_bias(n);
    const auto kernel = fused ? matmul::linear_relu_fused : matmul::linear_relu_unfused;

    for (auto _ : state) {
        kernel(A.data(), B.data(), bias.get(), C.data(), n, tile, threads);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.counters["FLOPS"] = benchmark::Counter(2.0 * n * n * n,
                                                 benchmark::Counter::kIsIterationInvariantRate);
}

// T = 64 for the small sizes, as in stage 4 (enough tiles for 6 threads), T = 128 otherwise.
BENCHMARK(BM_LinearRelu)
    ->ArgNames({"n", "T", "threads", "fused"})
    ->ArgsProduct({{256, 512}, {64}, {1, 6}, {0, 1}})
    ->ArgsProduct({{1000, 1024}, {128}, {1, 6}, {0, 1}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// The epilogue alone: bias + ReLU as two passes over C (passes:2) or as one (passes:1).
// Every pass reads and writes the whole matrix, 8 B per element. The sizes put C in
// L2 (256: 256 KiB), L3 (1024: 4 MiB) and DRAM (4096: 64 MiB).
// C changes in place from one iteration to the next (values drift by the bias); the
// amount of work does not depend on the values.
void BM_Epilogue(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int threads = static_cast<int>(state.range(1));
    const int passes = static_cast<int>(state.range(2));

    matmul::Matrix C(n);
    C.fill_random(kSeedA);
    const auto bias = random_bias(n);

    for (auto _ : state) {
        if (passes == 1) {
            matmul::add_bias_relu(C.data(), bias.get(), n, threads);
        } else {
            matmul::add_bias(C.data(), bias.get(), n, threads);
            matmul::relu(C.data(), n, threads);
        }
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    const double elements = static_cast<double>(n) * n;
    // 2 operations per element (an add and a max) in both variants.
    state.counters["FLOPS"] =
        benchmark::Counter(2.0 * elements, benchmark::Counter::kIsIterationInvariantRate);
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * passes *
                            static_cast<std::int64_t>(elements * 2 * sizeof(float)));
}

BENCHMARK(BM_Epilogue)
    ->ArgNames({"n", "threads", "passes"})
    ->ArgsProduct({{256, 1024, 4096}, {1, 6}, {2, 1}})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

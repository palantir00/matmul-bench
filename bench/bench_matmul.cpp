// Matrix multiplication benchmarks: one BENCHMARK_CAPTURE entry per kernel version.
#include <benchmark/benchmark.h>

#include "matmul/matmul.hpp"
#include "matmul/matrix.hpp"

namespace {

using Kernel = void (*)(const float*, const float*, float*, int);

// Same seeds as the tests: fixed input data on every run.
constexpr unsigned kSeedA = 42;
constexpr unsigned kSeedB = 43;

void BM_Matmul(benchmark::State& state, Kernel kernel) {
    const int n = static_cast<int>(state.range(0));
    matmul::Matrix A(n), B(n), C(n);
    A.fill_random(kSeedA);
    B.fill_random(kSeedB);
    // Touch C before timing: the first write to a fresh memory page triggers a page fault,
    // which must not end up inside the measured loop.
    C.fill(0.0f);

    for (auto _ : state) {
        kernel(A.data(), B.data(), C.data(), n);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    // n^2 dot products of length n, each n multiplies + n adds.
    const double flops = 2.0 * n * n * n;
    state.counters["FLOPS"] =
        benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

// The same sizes for every version. 1000 is a non-power-of-two control size: at 1024 the
// column stride of B is exactly 4 KiB, which causes cache set conflicts in the naive loop.
void Sizes(benchmark::internal::Benchmark* b) {
    for (int n : {256, 512, 1000, 1024}) {
        b->Arg(n);
    }
    b->Unit(benchmark::kMillisecond);
}

BENCHMARK_CAPTURE(BM_Matmul, naive, matmul::naive)->Apply(Sizes);
BENCHMARK_CAPTURE(BM_Matmul, ikj, matmul::ikj)->Apply(Sizes);

}  // namespace

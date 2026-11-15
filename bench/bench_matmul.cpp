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

BENCHMARK_CAPTURE(BM_Matmul, naive, matmul::naive)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1000)  // non-power-of-two control size (cache set conflicts at 1024)
    ->Arg(1024)
    ->Unit(benchmark::kMillisecond);

}  // namespace

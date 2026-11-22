// Memory-hierarchy benchmark: shows how throughput drops as the working set moves
// from L1 to L2, L3 and DRAM. In stage 5 these numbers serve as measured bandwidths
// for the roofline model.
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include "matmul/matrix.hpp"

namespace {

// y = a * x + y
// Per element: 2 FLOP (one multiply-add), 12 bytes of traffic (load x, load y, store y).
void saxpy(float a, const float* x, float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

void BM_Saxpy(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    // Cache-line aligned buffers. With 16-byte aligned std::vector storage the L1 case
    // varied ~1.5x between repetitions because of split stores (docs/NOTES.md, stage 0).
    auto x = matmul::alloc_aligned(n);
    auto y = matmul::alloc_aligned(n);
    std::fill(x.get(), x.get() + n, 1.0f);
    std::fill(y.get(), y.get() + n, 0.0f);
    const float a = 0.5f;

    for (auto _ : state) {
        saxpy(a, x.get(), y.get(), n);
        // Make the compiler believe y is used, so the loop cannot be removed.
        benchmark::DoNotOptimize(y.get());
        benchmark::ClobberMemory();
    }

    // kIsIterationInvariantRate: value is multiplied by the iteration count and
    // divided by elapsed time, i.e. reported as FLOP per second.
    state.counters["FLOPS"] = benchmark::Counter(
        2.0 * static_cast<double>(n), benchmark::Counter::kIsIterationInvariantRate);
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(n * 3 * sizeof(float)));
    state.SetLabel("working set " + std::to_string(2 * n * sizeof(float) / 1024) + " KiB");
}

// Working set = 2 arrays * n * 4 B. Sizes chosen against a P-core's caches.
BENCHMARK(BM_Saxpy)
    ->Arg(2 * 1024)           //  16 KiB -> fits L1d (48 KiB)
    ->Arg(64 * 1024)          // 512 KiB -> fits L2 (1.25 MiB)
    ->Arg(1024 * 1024)        //   8 MiB -> fits L3 (24 MiB)
    ->Arg(16 * 1024 * 1024);  // 128 MiB -> DRAM

// The same saxpy split over threads (each thread gets one contiguous chunk). On a DRAM-sized
// working set this measures how much memory bandwidth several cores can pull together,
// which is the "memory roof" for the roofline model in stage 5.
void saxpy_omp(float a, const float* x, float* y, std::ptrdiff_t n, int threads) {
#pragma omp parallel for schedule(static) num_threads(threads)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

void BM_SaxpyOmp(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const int threads = static_cast<int>(state.range(1));
    auto x = matmul::alloc_aligned(n);
    auto y = matmul::alloc_aligned(n);
    std::fill(x.get(), x.get() + n, 1.0f);
    std::fill(y.get(), y.get() + n, 0.0f);

    for (auto _ : state) {
        saxpy_omp(0.5f, x.get(), y.get(), static_cast<std::ptrdiff_t>(n), threads);
        benchmark::DoNotOptimize(y.get());
        benchmark::ClobberMemory();
    }

    state.counters["FLOPS"] = benchmark::Counter(
        2.0 * static_cast<double>(n), benchmark::Counter::kIsIterationInvariantRate);
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(n * 3 * sizeof(float)));
}

BENCHMARK(BM_SaxpyOmp)
    ->ArgNames({"n", "threads"})
    ->ArgsProduct({{16 * 1024 * 1024}, {1, 2, 4, 6, 8, 10, 16}})
    ->UseRealTime();  // by default CPU time would cover only the main thread

}  // namespace

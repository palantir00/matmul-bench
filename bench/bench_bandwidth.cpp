// Stage 0 smoke benchmark: checks that Google Benchmark works and shows how
// throughput drops as the working set moves from L1 to L2, L3 and DRAM.
// Later (stage 5) these numbers serve as measured bandwidths for the roofline model.
#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
    // NOTE: std::vector only guarantees 16-byte alignment. When y is not 32-byte aligned,
    // every second AVX store splits across two cache lines, so the L1-sized case varies
    // ~1.5x between repetitions (see docs/NOTES.md, stage 0). Stage 1 adds 64-byte aligned buffers.
    std::vector<float> x(n, 1.0f);
    std::vector<float> y(n, 0.0f);
    const float a = 0.5f;

    for (auto _ : state) {
        saxpy(a, x.data(), y.data(), n);
        // Make the compiler believe y is used, so the loop cannot be removed.
        benchmark::DoNotOptimize(y.data());
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

}  // namespace

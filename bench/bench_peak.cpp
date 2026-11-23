// Compute roof for the roofline model: the highest FP32 throughput the cores actually reach
// (at whatever clock the power limit allows), measured instead of taken from the datasheet.
//
// Each thread runs 12 independent chains of 256-bit FMAs. One chain alone is limited by
// the FMA latency (4 cycles, see clock_probe). A P-core has 2 FMA units, so it needs
// 2 x 4 = 8 chains in flight to keep both busy; 12 leave a margin and still fit in the
// 16 vector registers together with the two constants.
#include <benchmark/benchmark.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

namespace {

constexpr int kChains = 12;
constexpr long kStepsPerChunk = 1L << 15;  // ~0.07 ms on a P-core
constexpr int kChunksPerThread = 32;
// Per step: kChains FMAs x 8 floats x 2 FLOP.
constexpr double kFlopPerChunk = 2.0 * 8 * kChains * kStepsPerChunk;

void fma_chains(float a, float b) {
    const __m256 va = _mm256_set1_ps(a);
    const __m256 vb = _mm256_set1_ps(b);
    __m256 acc[kChains];
    for (__m256& x : acc) {
        x = _mm256_set1_ps(1.0f);
    }
    // acc = acc * a + b converges to b / (1 - a) = 1 for a = 0.999, b = 0.001: no overflow,
    // no denormals, so every FMA runs at full speed.
    for (long s = 0; s < kStepsPerChunk; ++s) {
#pragma GCC unroll 12
        for (int c = 0; c < kChains; ++c) {
            acc[c] = _mm256_fmadd_ps(acc[c], va, vb);
        }
    }
    benchmark::DoNotOptimize(acc);
}

// The work is split into many small chunks handed out dynamically. With a fixed share per
// thread, an iteration would last as long as the slowest thread (an E-core does 16 FLOP per
// cycle instead of 32, HT siblings share one core's FMA units), and we would measure
// "slowest core x threads" instead of the total throughput.
void BM_PeakFma(benchmark::State& state) {
    const int threads = static_cast<int>(state.range(0));
    const int chunks = kChunksPerThread * threads;
    for (auto _ : state) {
#pragma omp parallel for schedule(dynamic) num_threads(threads)
        for (int c = 0; c < chunks; ++c) {
            fma_chains(0.999f, 0.001f);
        }
    }
    state.counters["FLOPS"] = benchmark::Counter(kFlopPerChunk * chunks,
                                                 benchmark::Counter::kIsIterationInvariantRate);
}

// 6 = all P-cores, 10 = all physical cores, 16 = with HT (HT cannot add FMA units).
BENCHMARK(BM_PeakFma)
    ->ArgName("threads")
    ->Arg(1)
    ->Arg(6)
    ->Arg(10)
    ->Arg(16)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace

#endif  // __AVX2__ && __FMA__ (without -march=native there is no compute roof benchmark)

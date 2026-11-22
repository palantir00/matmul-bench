// Matrix multiplication benchmarks: one entry per kernel version, plus a tile size sweep.
#include <benchmark/benchmark.h>

#include "matmul/matmul.hpp"
#include "matmul/matrix.hpp"

namespace {

using Kernel = void (*)(const float*, const float*, float*, int);

// Same seeds as the tests: fixed input data on every run.
constexpr unsigned kSeedA = 42;
constexpr unsigned kSeedB = 43;

// Shared body: set up the matrices, time kernel(A, B, C, n), report GFLOP/s.
template <typename F>
void run_matmul(benchmark::State& state, int n, F kernel) {
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

void BM_Matmul(benchmark::State& state, Kernel kernel) {
    run_matmul(state, static_cast<int>(state.range(0)), kernel);
}

void BM_Tiled(benchmark::State& state) {
    const int tile = static_cast<int>(state.range(1));
    run_matmul(state, static_cast<int>(state.range(0)),
               [tile](const float* A, const float* B, float* C, int n) {
                   matmul::tiled(A, B, C, n, tile);
               });
}

void BM_TiledOmp(benchmark::State& state) {
    const int tile = static_cast<int>(state.range(1));
    const int threads = static_cast<int>(state.range(2));
    const auto schedule =
        state.range(3) != 0 ? matmul::Schedule::Dynamic : matmul::Schedule::Static;
    run_matmul(state, static_cast<int>(state.range(0)),
               [=](const float* A, const float* B, float* C, int n) {
                   matmul::tiled_omp(A, B, C, n, tile, threads, schedule);
               });
}

// The same sizes for every version. 1000 is a non-power-of-two control size: at 1024 the
// row stride is exactly 4 KiB, which causes cache set conflicts.
constexpr int kSizes[] = {256, 512, 1000, 1024};

// Rates (GFLOP/s) are computed from wall-clock time. By default Google Benchmark divides by
// CPU time, which leaves out moments when the OS took the core away, and which with
// OpenMP threads (stage 4) covers only the main thread.
void Common(benchmark::internal::Benchmark* b) {
    b->UseRealTime();
    b->Unit(benchmark::kMillisecond);
}

void Sizes(benchmark::internal::Benchmark* b) {
    for (int n : kSizes) {
        b->Arg(n);
    }
    Common(b);
}

// Every size against every tile size. A tile >= n is one tile, i.e. plain ikj.
void SizesAndTiles(benchmark::internal::Benchmark* b) {
    b->ArgNames({"n", "T"});
    for (int n : kSizes) {
        for (int tile : {16, 32, 64, 128, 256}) {
            b->Args({n, tile});
        }
    }
    Common(b);
}

// Thread scaling. 6 = all P-cores, 10 = all physical cores (P + E), 16 = with HT.
// Main case N = 1000/1024: static vs dynamic schedule with T = 128 (64 tiles), and dynamic
// with T = 64 (256 smaller tiles). Small sizes: a few thread counts only.
void OmpSweep(benchmark::internal::Benchmark* b) {
    b->ArgNames({"n", "T", "threads", "dyn"});
    for (int n : {1000, 1024}) {
        for (int threads : {1, 2, 4, 6, 8, 10, 12, 16}) {
            b->Args({n, 128, threads, 0});
            b->Args({n, 128, threads, 1});
            b->Args({n, 64, threads, 1});
        }
    }
    for (int n : {256, 512}) {
        for (int threads : {1, 6, 10, 16}) {
            b->Args({n, 64, threads, 1});
        }
    }
    Common(b);
}

BENCHMARK_CAPTURE(BM_Matmul, naive, matmul::naive)->Apply(Sizes);
BENCHMARK_CAPTURE(BM_Matmul, ikj, matmul::ikj)->Apply(Sizes);
BENCHMARK(BM_Tiled)->Apply(SizesAndTiles);
BENCHMARK(BM_TiledOmp)->Apply(OmpSweep);

}  // namespace

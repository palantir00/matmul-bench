// Estimates the core clock and the latency of scalar FP add and FMA, without access to
// hardware counters (WSL2 has none).
//
// Method: a chain of dependent integer adds (reg + reg) runs at exactly 1 instruction per
// cycle, so its duration per instruction is one clock cycle. Dividing the duration of
// another dependent chain by it gives that instruction's latency in cycles.
//
// Pitfall: "add $imm, reg" chains are partly executed by the register renamer on
// Golden Cove / Raptor Cove and run faster than 1 per cycle (a fake ~15 GHz clock).
// Register-register add and imul (3 cycles, used as a sanity check) are not affected.
//
// x86-64, GCC/Clang only (inline assembly). Run it right before a benchmark session:
// the clock depends on the power mode, temperature and on whether the OS put the
// thread on a P-core or an E-core.
#include <chrono>
#include <cstdint>
#include <cstdio>

#define R10(x) x x x x x x x x x x
#define R100(x) R10(R10(x))

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main() {
    constexpr long kIters = 10'000'000;      // x 100 dependent instructions per iteration
    constexpr double kOps = kIters * 100.0;  // = 1e9 instructions per chain

    for (int run = 0; run < 3; ++run) {
        std::uint64_t r = 0;
        std::uint64_t m = 1;
        const std::uint64_t one = 1;
        float f = 1.0f;
        const float tiny = 1e-8f;

        const auto t0 = Clock::now();
        for (long i = 0; i < kIters; ++i) asm volatile(R100("add %1, %0\n\t") : "+r"(r) : "r"(one));
        const auto t1 = Clock::now();
        for (long i = 0; i < kIters; ++i) asm volatile(R100("imul %1, %0\n\t") : "+r"(m) : "r"(one));
        const auto t2 = Clock::now();
        for (long i = 0; i < kIters; ++i) asm volatile(R100("vaddss %1, %0, %0\n\t") : "+x"(f) : "x"(tiny));
        const auto t3 = Clock::now();
        for (long i = 0; i < kIters; ++i) asm volatile(R100("vfmadd231ss %1, %1, %0\n\t") : "+x"(f) : "x"(tiny));
        const auto t4 = Clock::now();

        const double cycle = seconds(t0, t1);  // duration of the 1-cycle chain
        std::printf("run %d: clock ~%.2f GHz | imul %.2f cycles (expect 3) | vaddss %.2f | vfmadd231ss %.2f\n",
                    run, kOps / cycle / 1e9, seconds(t1, t2) / cycle, seconds(t2, t3) / cycle,
                    seconds(t3, t4) / cycle);
    }
    return 0;
}

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <random>

namespace matmul {

// Cache line size on x86: the unit in which data moves between memory and the caches.
inline constexpr std::size_t kCacheLine = 64;

struct FreeDeleter {
    void operator()(float* p) const { std::free(p); }
};
using AlignedFloats = std::unique_ptr<float[], FreeDeleter>;

// Allocates `count` floats starting at a cache line boundary, so 32-byte AVX loads and
// stores never straddle two lines (see docs/NOTES.md, stage 0: split stores).
inline AlignedFloats alloc_aligned(std::size_t count) {
    // std::aligned_alloc requires the size to be a multiple of the alignment.
    std::size_t bytes = std::max<std::size_t>(count * sizeof(float), 1);
    bytes = (bytes + kCacheLine - 1) / kCacheLine * kCacheLine;
    void* p = std::aligned_alloc(kCacheLine, bytes);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return AlignedFloats(static_cast<float*>(p));
}

// Owning n x n row-major float matrix with cache-line aligned storage.
// Used by tests and benchmarks only; the kernels take raw pointers (BLAS style).
class Matrix {
public:
    explicit Matrix(int n) : n_(n), data_(alloc_aligned(static_cast<std::size_t>(n) * n)) {}

    int n() const { return n_; }
    std::size_t size() const { return static_cast<std::size_t>(n_) * n_; }
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }

    float& operator()(int i, int j) { return data_[static_cast<std::size_t>(i) * n_ + j]; }
    float operator()(int i, int j) const { return data_[static_cast<std::size_t>(i) * n_ + j]; }

    void fill(float value) { std::fill(data(), data() + size(), value); }

    // Uniform values in [-1, 1) from a fixed seed: identical data on every run.
    void fill_random(unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::generate(data(), data() + size(), [&] { return dist(gen); });
    }

private:
    int n_;
    AlignedFloats data_;
};

}  // namespace matmul

#!/usr/bin/env python3
"""PyTorch (BLAS) reference for the same problems as the C++ benchmarks.

Measures, for square float32 matrices:
  matmul       C = A @ B into a preallocated C (like our kernels)
  linear_relu  C = relu(A @ B + bias): addmm (bias folded into the BLAS call) + in-place ReLU,
               i.e. what eager PyTorch runs for nn.Linear + ReLU

Method as in scripts/run_bench.sh: fixed seed, warm-up, every repetition times enough calls
to last --min-time, repetitions of all configurations are interleaved in random order,
median and coefficient of variation over repetitions. Writes JSON to --out.
"""
import argparse
import json
import platform
import random
import statistics
import time

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1000, 1024])
    p.add_argument("--threads", type=int, nargs="+", default=[1, 6],
                   help="thread counts for every size and both ops")
    p.add_argument("--sweep-n", type=int, default=1024, help="size for the thread sweep of matmul")
    p.add_argument("--sweep-threads", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 12, 16])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--min-time", type=float, default=0.5, help="seconds per repetition")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


def make_inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(n, n, generator=g) * 2 - 1  # uniform [-1, 1), like Matrix::fill_random
    B = torch.rand(n, n, generator=g) * 2 - 1
    bias = torch.rand(n, generator=g) * 2 - 1
    return A, B, bias


def make_op(op, A, B, bias, C):
    if op == "matmul":
        return lambda: torch.matmul(A, B, out=C)
    if op == "linear_relu":
        return lambda: torch.relu_(torch.addmm(bias, A, B, out=C))
    raise ValueError(op)


def accuracy(A, B, C):
    """Largest |C - exact| / (|A| @ |B|), the same error measure as the C++ tests."""
    exact = A.double() @ B.double()
    scale = A.double().abs() @ B.double().abs()
    return ((C.double() - exact).abs() / scale).max().item()


def linear_kernels(n):
    """Names of the ATen operators that F.linear + ReLU actually run (checks the stage 5 claim)."""
    A, B, bias = make_inputs(n, 0)
    W = B.t()  # nn.Linear stores weights as (out, in) and computes x @ W.T
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        torch.relu(F.linear(A, W, bias))
    return [e.key for e in prof.key_averages()]


def main():
    args = parse_args()
    configs = [(op, n, t) for op in ("matmul", "linear_relu") for n in args.sizes for t in args.threads]
    configs += [("matmul", args.sweep_n, t) for t in args.sweep_threads
                if ("matmul", args.sweep_n, t) not in configs]

    data = {}
    for n in sorted({n for _, n, _ in configs}):
        A, B, bias = make_inputs(n, args.seed)
        data[n] = (A, B, bias, torch.empty(n, n))

    # Calibration: calls per repetition so that one repetition lasts about --min-time.
    iters = {}
    for op, n, t in configs:
        torch.set_num_threads(t)
        fn = make_op(op, *data[n])
        fn()  # warm-up: thread pool, BLAS kernel selection, first touch of C
        t0 = time.perf_counter()
        fn()
        one = time.perf_counter() - t0
        iters[(op, n, t)] = max(1, int(args.min_time / max(one, 1e-7)))

    times = {c: [] for c in configs}
    rng = random.Random(args.seed)
    for rep in range(args.reps):
        order = configs[:]
        rng.shuffle(order)  # interleaving: slow drifts (clock, background) hit all configs alike
        for c in order:
            op, n, t = c
            torch.set_num_threads(t)
            fn = make_op(op, *data[n])
            fn()  # warm-up after the thread count change
            k = iters[c]
            t0 = time.perf_counter()
            for _ in range(k):
                fn()
            times[c].append((time.perf_counter() - t0) / k)
        print(f"repetition {rep + 1}/{args.reps} done", flush=True)

    results = []
    for c in configs:
        op, n, t = c
        ts = times[c]
        med = statistics.median(ts)
        results.append({
            "op": op, "n": n, "threads": t, "reps": len(ts), "iterations": iters[c],
            "median_s": med, "cv": statistics.stdev(ts) / statistics.mean(ts) if len(ts) > 1 else 0.0,
            "gflops": 2.0 * n ** 3 / med / 1e9,  # FLOP count of the matmul only, as in C++
        })
        print(f"{op:12} n={n:5} threads={t:2}  {med * 1e3:9.3f} ms  "
              f"{results[-1]['gflops']:7.1f} GFLOP/s  cv={results[-1]['cv'] * 100:4.1f}%")

    torch.set_num_threads(1)
    acc = {}
    for n in args.sizes:
        A, B, _, C = data[n]
        torch.matmul(A, B, out=C)
        acc[n] = accuracy(A, B, C)

    config_lines = []
    for line in torch.__config__.show().splitlines():
        line = line.strip()
        if "Build settings" in line:  # one very long line: keep only the relevant fields
            config_lines += [f.strip() for f in line.split(",")
                             if f.strip().startswith(("BLAS_INFO", "USE_MKL", "USE_MKLDNN", "USE_OPENMP"))]
        elif any(k in line for k in ("Intel(R)", "OpenMP", "CPU capability")):
            config_lines.append(line)
    out = {
        "context": {
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "mkl": torch.backends.mkl.is_available(),
            "config": config_lines,
            "linear_relu_aten_ops": linear_kernels(256),
            "max_error_over_scale": acc,  # compare with the n * eps bound from the C++ tests
            "reps": args.reps, "min_time_s": args.min_time, "seed": args.seed,
        },
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("aten ops of F.linear + relu:", out["context"]["linear_relu_aten_ops"])
    print("max |error| / scale:", {n: f"{e:.2e}" for n, e in acc.items()})
    print("wrote", args.out)


if __name__ == "__main__":
    main()

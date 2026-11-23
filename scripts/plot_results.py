#!/usr/bin/env python3
"""Plots and README tables from the final measurement series.

Reads results/final_cpp.json (Google Benchmark) and results/final_torch.json
(scripts/bench_torch.py), writes PNG charts to docs/img/ and prints the Markdown tables
used in README.md, so every number in the README can be regenerated.

Usage: python scripts/plot_results.py [--results results] [--img docs/img]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

# Light-mode chart palette: surface, inks, hairlines, and categorical slots in fixed order
# (slot 1 = my code / 1 thread, slot 2 = PyTorch / 6 threads, ...).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 7,
})

SIZES = [256, 512, 1000, 1024]
THREADS = [1, 2, 4, 6, 8, 10, 12, 16]


class Cpp:
    """Medians and CVs from a Google Benchmark JSON file, by run name."""

    def __init__(self, path):
        with open(path) as f:
            d = json.load(f)
        self.context = d["context"]
        self.med, self.cv = {}, {}
        for b in d["benchmarks"]:
            if b.get("run_type") != "aggregate":
                continue
            if b["aggregate_name"] == "median":
                self.med[b["run_name"]] = b
            elif b["aggregate_name"] == "cv":
                self.cv[b["run_name"]] = b

    def gflops(self, name):
        return self.med[name]["FLOPS"] / 1e9

    def gbps(self, name):
        return self.med[name]["bytes_per_second"] / 1e9

    def ms(self, name):
        b = self.med[name]
        scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}[b["time_unit"]]
        return b["real_time"] * scale

    def cv_pct(self, name):
        return 100.0 * self.cv[name]["real_time"]


def naive(n):
    return f"BM_Matmul/naive/{n}/real_time"


def ikj(n):
    return f"BM_Matmul/ikj/{n}/real_time"


def tiled(n, t=128):
    return f"BM_Tiled/n:{n}/T:{t}/real_time"


def omp(n, p, t=None):
    t = t or (64 if n <= 512 else 128)  # the tile sizes measured in the thread sweep
    return f"BM_TiledOmp/n:{n}/T:{t}/threads:{p}/dyn:1/real_time"


def linear(n, p, fused):
    t = 64 if n <= 512 else 128
    return f"BM_LinearRelu/n:{n}/T:{t}/threads:{p}/fused:{int(fused)}/real_time"


def epilogue(n, p, passes):
    return f"BM_Epilogue/n:{n}/threads:{p}/passes:{passes}/real_time"


def peak(p):
    return f"BM_PeakFma/threads:{p}/real_time"


def saxpy_omp(p):
    return f"BM_SaxpyOmp/n:16777216/threads:{p}/real_time"


def load_torch(path):
    with open(path) as f:
        d = json.load(f)
    return d["context"], {(r["op"], r["n"], r["threads"]): r for r in d["results"]}


def fmt(x):
    return f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}"


def title(ax, text, sub):
    ax.set_title(text, loc="left", fontsize=12, color=INK, pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


# ---- Charts ---------------------------------------------------------------------------

def chart_versions(cpp, torch, path, n=1024):
    rows = [  # (label, GFLOP/s, threads)
        ("naive (i-j-k)", cpp.gflops(naive(n)), 1),
        ("i-k-j loop order", cpp.gflops(ikj(n)), 1),
        ("tiled, T = 128", cpp.gflops(tiled(n)), 1),
        ("tiled + OpenMP", cpp.gflops(omp(n, 6)), 6),
        ("PyTorch (MKL)", torch[("matmul", n, 1)]["gflops"], 1),
        ("PyTorch (MKL)", torch[("matmul", n, 6)]["gflops"], 6),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    ys = list(range(len(rows)))[::-1]
    for y, (label, g, p) in zip(ys, rows):
        ax.plot([g], [y], "o", color=SLOTS[0] if p == 1 else SLOTS[1], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(g * 1.18, y, f"{fmt(g)}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.axhline(1.5, color=AXIS, linewidth=0.8)  # my versions above, the BLAS reference below
    for value, label in [(cpp.gflops(peak(1)), "FMA peak, 1 core"), (cpp.gflops(peak(6)), "FMA peak, 6 cores")]:
        ax.axvline(value, color=INK2, linewidth=1.0)
        ax.text(value, len(rows) - 0.45, f"{label}\n{value:.0f}", ha="center", va="bottom",
                fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 1500)
    ax.set_ylim(-0.6, len(rows) - 0.4 + 0.9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("GFLOP/s (log scale, higher is better)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.plot([], [], "o", color=SLOTS[0], label="1 thread")
    ax.plot([], [], "o", color=SLOTS[1], label="6 threads")
    ax.legend(loc="lower left", fontsize=9)
    title(ax, f"Matrix multiplication, N = {n}, float32",
          "median of repetitions; i7-13620H laptop, WSL2")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_threads(cpp, torch, path, n=1024):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    mine = [cpp.gflops(omp(n, p)) for p in THREADS]
    mkl = [torch[("matmul", n, p)]["gflops"] for p in THREADS]
    peak_threads = [1, 6, 10, 16]
    peaks = [cpp.gflops(peak(p)) for p in peak_threads]
    # The FMA peak was measured at 4 thread counts only: points without a connecting line,
    # so the chart does not suggest values in between.
    ax.plot(peak_threads, peaks, "o", color=SLOTS[2], label="FMA peak (measured)", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(16.3, peaks[-1], f"FMA peak\n{fmt(peaks[-1])}", va="center", fontsize=8, color=INK2)
    for label, ys, color in [("PyTorch (MKL)", mkl, SLOTS[1]), ("tiled + OpenMP (my code)", mine, SLOTS[0])]:
        ax.plot(THREADS, ys, "-o", color=color, label=label, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(16.3, ys[-1], f"{label}\n{fmt(ys[-1])}", va="center", fontsize=8, color=INK2)
    for x, text in [(6, "6 P-cores"), (10, "+ 4 E-cores")]:
        ax.axvline(x, color=AXIS, linewidth=1.0, zorder=1)
        ax.text(x + 0.15, 12, text, fontsize=8, color=INK2, va="bottom")
    ax.set_xticks(THREADS)
    ax.set_xlim(0.5, 19.5)
    ax.set_ylim(0, None)
    ax.set_xlabel("threads")
    ax.set_ylabel("GFLOP/s")
    ax.legend(loc="upper left", fontsize=9)
    title(ax, f"Thread scaling, N = {n}",
          "beyond 6 threads Windows uses hyperthreads and the power limit lowers the clock")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_tiles(cpp, path):
    tiles = [16, 32, 64, 128, 256]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    best = []
    for color, n in zip(SLOTS, SIZES):
        ys = [cpp.gflops(tiled(n, t)) for t in tiles]
        ax.plot(tiles, ys, "-o", color=color, label=f"N = {n}", markeredgecolor=SURFACE, markeredgewidth=2)
        best.append(f"N = {n}: {fmt(ys[tiles.index(128)])}")
    # The curves cross and meet near T = 256, so the values at the best tile go into one note.
    ax.text(15, 31.5, "at T = 128 (best for every N):\n" + "\n".join(best), fontsize=8, color=INK2, va="top")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(tiles))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(13, 360)
    ax.set_ylim(0, None)
    ax.set_xlabel("tile size T (floats)")
    ax.set_ylabel("GFLOP/s, 1 thread")
    ax.legend(loc="lower right", fontsize=9)
    title(ax, "Tiling: GFLOP/s against tile size",
          "small tiles pay loop overhead, large tiles stop fitting in cache")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_roofline(cpp, torch, path, n=1024):
    peak6 = cpp.gflops(peak(6))
    bw = cpp.gbps(saxpy_omp(6))
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [0.05 * 1.1 ** i for i in range(120)]
    xs = [x for x in xs if x <= 1000]
    ax.plot(xs, [min(peak6, bw * x) for x in xs], color=INK2, linewidth=1.5, zorder=2)
    ridge = peak6 / bw
    ax.text(0.6, bw * 0.6 * 2.2, f"DRAM bandwidth\n{bw:.0f} GB/s × AI", fontsize=8, color=INK2, ha="right")
    ax.text(300, peak6 * 1.12, f"FMA peak, 6 threads: {peak6:.0f} GFLOP/s", ha="center", fontsize=8, color=INK2)
    ax.axvline(ridge, color=AXIS, linewidth=0.8, zorder=1)
    ax.text(ridge * 1.08, 1.3, f"ridge point\n{ridge:.0f} FLOP/B", fontsize=8, color=INK2)

    points = [  # (label, AI, GFLOP/s, slot, label offset factors)
        ("bias + ReLU, 2 passes", 2 / 16, cpp.gflops(epilogue(4096, 6, 2)), 0, (1.15, 0.8)),
        ("bias + ReLU, fused", 2 / 8, cpp.gflops(epilogue(4096, 6, 1)), 0, (1.15, 0.9)),
        ("matmul, my code", n / 6, cpp.gflops(omp(n, 6)), 0, (1.15, 0.85)),
        ("matmul, PyTorch (MKL)", n / 6, torch[("matmul", n, 6)]["gflops"], 1, (1.15, 0.95)),
    ]
    for label, ai, g, slot, (dx, dy) in points:
        ax.plot([ai], [g], "o", color=SLOTS[slot], markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        ax.text(ai * dx, g * dy, f"{label}\n{fmt(g)} GFLOP/s", fontsize=8, color=INK2, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 1000)
    ax.set_ylim(1, 1500)
    fmt_axis = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt_axis)
    ax.yaxis.set_major_formatter(fmt_axis)
    ax.set_xlabel("arithmetic intensity (FLOP per byte of DRAM traffic, log)")
    ax.set_ylabel("GFLOP/s (log)")
    ax.plot([], [], "o", color=SLOTS[0], label="my kernels")
    ax.plot([], [], "o", color=SLOTS[1], label="PyTorch (MKL)")
    ax.legend(loc="upper left", fontsize=9)
    title(ax, "Roofline, 6 threads (measured roofs)",
          f"matmul AI = N/6 for N = {n} (each matrix moved once); epilogue on a 4096 x 4096 matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---- Tables -----------------------------------------------------------------------------

def tables(cpp, torch, tctx):
    p1, p6 = cpp.gflops(peak(1)), cpp.gflops(peak(6))
    print("### GFLOP/s by version (median)\n")
    print("| version | threads | N=256 | N=512 | N=1000 | N=1024 | vs naive (1024) | % of FMA peak (1024) |")
    print("|---|---|---|---|---|---|---|---|")
    base = cpp.gflops(naive(1024))
    rows = [
        ("naive (i-j-k)", 1, [cpp.gflops(naive(n)) for n in SIZES]),
        ("i-k-j", 1, [cpp.gflops(ikj(n)) for n in SIZES]),
        ("tiled, T=128", 1, [cpp.gflops(tiled(n)) for n in SIZES]),
        ("tiled + OpenMP", 6, [cpp.gflops(omp(n, 6)) for n in SIZES]),
        ("PyTorch (MKL)", 1, [torch[("matmul", n, 1)]["gflops"] for n in SIZES]),
        ("PyTorch (MKL)", 6, [torch[("matmul", n, 6)]["gflops"] for n in SIZES]),
    ]
    for label, p, gs in rows:
        pk = p1 if p == 1 else p6
        print(f"| {label} | {p} | " + " | ".join(fmt(g) for g in gs) +
              f" | {gs[-1] / base:.0f}x | {100 * gs[-1] / pk:.0f}% |")
    print(f"\nFMA peak (BM_PeakFma): 1 thread {p1:.1f}, 6 threads {p6:.0f} GFLOP/s.")
    worst_cv = max(cpp.cv_pct(name) for name in [naive(1024), ikj(1024), tiled(1024), omp(1024, 6)])
    print(f"Worst CV in the C++ rows at N=1024: {worst_cv:.1f}%.")

    print("\n### Thread scaling, N=1024 (GFLOP/s, median; CV in brackets)\n")
    print("| threads | " + " | ".join(str(p) for p in THREADS) + " |")
    print("|---|" + "---|" * len(THREADS))
    print("| tiled + OpenMP | " + " | ".join(f"{fmt(cpp.gflops(omp(1024, p)))} ({cpp.cv_pct(omp(1024, p)):.0f}%)"
                                          for p in THREADS) + " |")
    print("| PyTorch (MKL) | " + " | ".join(f"{fmt(torch[('matmul', 1024, p)]['gflops'])} "
                                         f"({100 * torch[('matmul', 1024, p)]['cv']:.0f}%)" for p in THREADS) + " |")
    print("| FMA peak | " + " | ".join(fmt(cpp.gflops(peak(p))) if peak(p) in cpp.med else "–"
                                     for p in THREADS) + " |")

    print("\n### Fusion\n")
    print("| case | threads | separate (CV) | fused (CV) | separate / fused |")
    print("|---|---|---|---|---|")
    cases = [(f"relu(A·B + bias), N=1024", p, linear(1024, p, False), linear(1024, p, True)) for p in (1, 6)]
    cases += [(f"bias + ReLU alone, 4096² in DRAM", p, epilogue(4096, p, 2), epilogue(4096, p, 1)) for p in (1, 6)]
    for label, p, sep, fus in cases:
        a, b = cpp.ms(sep), cpp.ms(fus)
        print(f"| {label} | {p} | {a:.2f} ms ({cpp.cv_pct(sep):.0f}%) | {b:.2f} ms ({cpp.cv_pct(fus):.0f}%) "
              f"| {a / b:.2f} |")
    for p in (1, 6):
        m, lr = torch[("matmul", 1024, p)]["median_s"] * 1e3, torch[("linear_relu", 1024, p)]["median_s"] * 1e3
        print(f"| PyTorch: matmul vs addmm + relu, N=1024 | {p} | – | – | matmul {m:.2f} ms, addmm+relu {lr:.2f} ms |")

    print("\n### Context\n")
    print(f"- C++ run: {cpp.context['date']}, commit {cpp.context.get('commit', '?')}")
    print(f"- PyTorch {tctx['torch']}, CPU capability {tctx['cpu_capability']}, " + "; ".join(tctx["config"]))
    print(f"- PyTorch max |error|/scale: {tctx['max_error_over_scale']}")
    print(f"- F.linear + relu runs: {tctx['linear_relu_aten_ops']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--img", default="docs/img")
    args = ap.parse_args()
    cpp = Cpp(os.path.join(args.results, "final_cpp.json"))
    tctx, torch = load_torch(os.path.join(args.results, "final_torch.json"))
    os.makedirs(args.img, exist_ok=True)
    chart_versions(cpp, torch, os.path.join(args.img, "versions.png"))
    chart_threads(cpp, torch, os.path.join(args.img, "threads.png"))
    chart_tiles(cpp, os.path.join(args.img, "tiles.png"))
    chart_roofline(cpp, torch, os.path.join(args.img, "roofline.png"))
    tables(cpp, torch, tctx)


if __name__ == "__main__":
    main()

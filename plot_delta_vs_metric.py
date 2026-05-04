"""
Plot a metric vs. delta for each benchmark, one line per benchmark.

Usage:
    python plot_delta_vs_metric.py                   # defaults to NMAE (float)
    python plot_delta_vs_metric.py --metric r2
    python plot_delta_vs_metric.py --metric nmae_fp
    python plot_delta_vs_metric.py --metric dnmae
    python plot_delta_vs_metric.py --metric pred_time
    python plot_delta_vs_metric.py --metric model_size

Available --metric choices (case-insensitive):
    nmae        → NMAE (float)
    r2          → R² (float)
    nmae_fp     → NMAE (FP C)
    dnmae       → ΔNMAE (float → FP C)
    pred_time   → Median Pred. Time (FP C)   [strips " us/sample"]
    model_size  → Model Size (FP C)          [strips " KB"]
"""

import argparse
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16

SPEC_DIR = "results/spec"
DELTAS = [0.01, 0.02, 0.03, 0.04, 0.05]
TARGETS = ["rem_time", "next_insn"]

METRICS = {
    "nmae":       dict(col=1, label="NMAE (float)",             ylabel="NMAE",                  strip_re=None,             ymin=0),
    "r2":         dict(col=2, label="R² (float)",               ylabel="R²",                    strip_re=None,             ymin=None),
    "nmae_fp":    dict(col=3, label="NMAE (FP C)",              ylabel="NMAE",                  strip_re=None,             ymin=0),
    "dnmae":      dict(col=4, label="ΔNMAE (float → FP C)",     ylabel="ΔNMAE",                 strip_re=None,             ymin=None),
    "pred_time":  dict(col=5, label="Median Pred. Time (FP C)", ylabel="Pred. time (µs/sample)",strip_re=r"\s*us/sample",  ymin=0),
    "model_size": dict(col=6, label="Model Size (FP C)",        ylabel="Model size (KB)",       strip_re=r"\s*KB",         ymin=0),
}

# Cycle through distinct colors and line styles for readability in B&W and color
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def parse_results(path, col, strip_re):
    values = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("Target"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > col:
                target = parts[0]
                raw = parts[col]
                if strip_re:
                    raw = re.sub(strip_re, "", raw).strip()
                try:
                    values[target] = float(raw)
                except ValueError:
                    pass
    return values


def build_data(metric_cfg):
    benchmarks = sorted(
        d for d in os.listdir(SPEC_DIR)
        if os.path.isdir(os.path.join(SPEC_DIR, d))
    )
    col = metric_cfg["col"]
    strip_re = metric_cfg["strip_re"]

    data = {t: np.full((len(benchmarks), len(DELTAS)), np.nan) for t in TARGETS}
    for bi, benchmark in enumerate(benchmarks):
        for di, delta in enumerate(DELTAS):
            path = os.path.join(SPEC_DIR, benchmark, f"delta={delta:.2f}", "results.txt")
            if not os.path.exists(path):
                continue
            parsed = parse_results(path, col, strip_re)
            for target in TARGETS:
                if target in parsed:
                    data[target][bi, di] = parsed[target]
    return benchmarks, data


def plot(benchmarks, data, metric_cfg, out_path):
    ylabel = metric_cfg["ylabel"]
    base, ext = os.path.splitext(out_path)

    for target in TARGETS:
        arr = data[target]
        fig, ax = plt.subplots(figsize=(7, 5))

        for bi, benchmark in enumerate(benchmarks):
            color = COLORS[bi % len(COLORS)]
            ls = LINESTYLES[bi % len(LINESTYLES)]
            marker = MARKERS[bi % len(MARKERS)]
            deltas_us = [d * 1000 for d in DELTAS]
            ax.plot(deltas_us, arr[bi], color=color, linestyle=ls, marker=marker,
                    markersize=6, linewidth=2.5, label=benchmark)

        ax.set_xlabel("Δ (ms)", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_xticks(deltas_us)
        ax.set_xticklabels([f"{d:.0f}" for d in deltas_us], fontsize=16)
        ax.tick_params(axis="y", labelsize=16)
        legend = ax.legend(fontsize=16, framealpha=0.7, loc="upper left")
        for text in legend.get_texts():
            text.set_fontstyle("italic")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        _, top = ax.get_ylim()
        ax.set_ylim(bottom=metric_cfg["ymin"], top=top * 1.15)

        target_out = f"{base}_{target}{ext}"
        plt.tight_layout()
        plt.savefig(target_out, bbox_inches="tight")
        plt.close()
        print(f"Saved {target_out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default="nmae",
                        help="Metric to plot. Choices: " + ", ".join(METRICS))
    parser.add_argument("--out", default=None,
                        help="Output file path (default: lineplot_<metric>.pdf)")
    args = parser.parse_args()

    key = args.metric.lower()
    if key not in METRICS:
        print(f"Unknown metric '{args.metric}'. Choose from: {', '.join(METRICS)}", file=sys.stderr)
        sys.exit(1)

    metric_cfg = METRICS[key]
    out_path = args.out or os.path.join("results", f"lineplot_{key}.pdf")

    benchmarks, data = build_data(metric_cfg)
    plot(benchmarks, data, metric_cfg, out_path)


if __name__ == "__main__":
    main()

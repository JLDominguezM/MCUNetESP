#!/usr/bin/env python3
"""
Genera todos los gráficos PNG del experimento MCUNet en ESP32-S3.

Inputs: latencias medidas en hardware (hardcoded abajo desde docs/findings.md).
Outputs: docs/plots/*.png
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# (modelo, tarea, input_w, input_h, dtype, latencia_ms, tagged_overhead_ms)
RESULTS = [
    ("mcunet-vww2",       "VWW 2-class",         144, 144, "int8",   3391.7, None),
    ("mcunet-in2",        "ImageNet 1000",       160, 160, "int8",   3916.8,   1.44),
    ("mcunet-person-det", "Detector bbox",       128, 160, "float32",1240.8, None),
    ("mbv2-w0.35",        "ImageNet 1000",       144, 144, "int8",   1644.5,   0.74),
    ("proxyless-w0.3",    "ImageNet 1000",       176, 176, "int8",   2511.4, None),
]

# Distribución de kernel sizes en DWCONVs
KERNELS = {
    "mcunet-in2":  {"3x3": 6,  "5x5": 9, "7x7": 3},
    "mbv2-w0.35":  {"3x3": 17, "5x5": 0, "7x7": 0},
}

COLOR_MCU = "#d62728"
COLOR_MBV = "#1f77b4"
COLOR_PROX = "#2ca02c"
COLOR_DET = "#9467bd"


def color_for(model: str) -> str:
    if "mcunet" in model and "person-det" in model: return COLOR_DET
    if "mcunet" in model: return COLOR_MCU
    if "mbv2" in model:   return COLOR_MBV
    if "proxyless" in model: return COLOR_PROX
    return "#7f7f7f"


def plot_latency_absolute():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [r[0] for r in RESULTS]
    lats   = [r[5] for r in RESULTS]
    colors = [color_for(m) for m in labels]
    bars = ax.bar(range(len(labels)), lats, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, lats):
        ax.text(b.get_x() + b.get_width()/2, v + 80, f"{v:.0f} ms",
                ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Latencia por inferencia (ms)", fontsize=11)
    ax.set_title("Latencia en XIAO ESP32-S3 Sense @ 240 MHz, esp-tflite-micro + esp-nn",
                 fontsize=11)
    ax.set_ylim(0, max(lats) * 1.18)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "01_latency_absolute.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT/'01_latency_absolute.png'}")


def plot_latency_per_pixel():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [r[0] for r in RESULTS]
    pxs    = [r[2]*r[3] for r in RESULTS]
    lats   = [r[5] for r in RESULTS]
    norm   = [l / p * 1000 for l, p in zip(lats, pxs)]  # µs/pixel
    colors = [color_for(m) for m in labels]
    bars = ax.bar(range(len(labels)), norm, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, norm):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f} µs/px",
                ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Latencia normalizada (µs / pixel de entrada)", fontsize=11)
    ax.set_title("Eficiencia por pixel — MCUNet vs baselines en ESP32-S3", fontsize=11)
    ax.set_ylim(0, max(norm) * 1.18)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "02_latency_per_pixel.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT/'02_latency_per_pixel.png'}")


def plot_kernel_distribution():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    models = list(KERNELS.keys())
    ksizes = ["3x3", "5x5", "7x7"]
    x = np.arange(len(ksizes))
    w = 0.35
    for i, m in enumerate(models):
        vals = [KERNELS[m][k] for k in ksizes]
        ax.bar(x + i*w, vals, w, label=m, edgecolor="black", linewidth=0.5,
               color=color_for(m))
        for xi, v in zip(x + i*w, vals):
            ax.text(xi, v + 0.3, str(v), ha="center", fontsize=10)
    ax.set_xticks(x + w/2)
    ax.set_xticklabels(ksizes)
    ax.set_xlabel("Tamaño de kernel DWCONV")
    ax.set_ylabel("Cantidad de ops")
    ax.set_title("Diversidad de kernels DWCONV — MCUNet vs MobileNetV2")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    # Annotate the cost of larger kernels
    ax.text(0.98, 0.92,
            "Costo relativo:\n  3×3 = 1.0×\n  5×5 = 2.78×\n  7×7 = 5.44×",
            transform=ax.transAxes, fontsize=9,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff7e6", ec="#cca96f"))
    fig.tight_layout()
    fig.savefig(OUT / "03_kernel_distribution.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT/'03_kernel_distribution.png'}")


def plot_op_breakdown():
    # Profiler-tagged ops vs invisible CONV/DWCONV
    fig, ax = plt.subplots(figsize=(9, 4))
    models = ["mcunet-in2", "mbv2-w0.35"]
    pad = [323761/240/1000, 172094/240/1000]  # ms (ticks/240MHz/1000)
    add = [21662/240/1000,  6124/240/1000]
    other = [41/240/1000, 40/240/1000]
    total = [3916.807, 1644.547]
    conv = [t - p - a - o for t, p, a, o in zip(total, pad, add, other)]

    y = np.arange(len(models))
    ax.barh(y, conv, color="#ff7f0e", edgecolor="black", linewidth=0.5,
            label="CONV / DWCONV (vía esp-nn, no tagged)")
    ax.barh(y, pad, left=conv, color="#1f77b4", edgecolor="black", linewidth=0.5,
            label="PAD")
    ax.barh(y, add, left=[c+p for c, p in zip(conv, pad)],
            color="#2ca02c", edgecolor="black", linewidth=0.5, label="ADD")
    for i, t in enumerate(total):
        ax.text(t + 50, i, f"{t:.0f} ms", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(models)
    ax.set_xlabel("Tiempo (ms)")
    ax.set_title("Desglose por op (datos: MicroProfiler) — CONV/DWCONV domina ambos")
    ax.legend(loc="lower right")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "04_op_breakdown.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT/'04_op_breakdown.png'}")


if __name__ == "__main__":
    plot_latency_absolute()
    plot_latency_per_pixel()
    plot_kernel_distribution()
    plot_op_breakdown()
    print("\nGenerated 4 plots in", OUT)

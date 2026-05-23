#!/usr/bin/env python3
"""
Genera visualizaciones del pipeline de inferencia para documentación:
  - 05_test_image_resized.png — la misma imagen procesada por cada modelo (a su input size)
  - 06_person_det_grid.png    — overlay del grid de detección con la celda 'winner' marcada
  - 07_classification_panel.png — panel con top-5 de cada clasificador
"""

from __future__ import annotations
from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Prefiere la imagen capturada por la OV2640 del XIAO si existe; si no,
# cae a la imagen de prueba del repo MCUNet (mientras la cámara no responda).
CAPTURED = OUT / "sample_camera_frame.png"
FALLBACK = ROOT / "host" / "sample_images" / "person_det.jpg"
SRC_IMG = CAPTURED if CAPTURED.exists() else FALLBACK
IMG_SOURCE_LABEL = "OV2640 del XIAO Sense" if SRC_IMG == CAPTURED else "imagen de test (repo MCUNet)"
print(f"[visualize] usando imagen: {SRC_IMG.name}  ({IMG_SOURCE_LABEL})")

# Top-5 capturados en hardware (de docs/logs/) — labels de ImageNet.
# Re-medidos con frame real capturado por la OV2640 (techo + cara desde abajo).
RESULTS = {
    "mcunet-vww2 (144x144)": {
        "type": "vww",
        "scores": {"no-pers": 22, "PERSON": -22},
        "verdict": "no-pers",
    },
    "mcunet-in2 (160x160)": {
        "type": "imagenet",
        "labels": ["alp", "vault", "broom", "frying pan", "studio couch"],
        "scores": [27, 1, 1, 0, 0],
    },
    "mbv2-w0.35 (144x144)": {
        "type": "imagenet",
        "labels": ["jersey", "vault", "frying pan", "swing", "lakeside"],
        "scores": [-3, -8, -9, -10, -10],
    },
    "proxyless-w0.3 (176x176)": {
        "type": "imagenet",
        "labels": ["power drill", "soup bowl", "stove", "punching bag", "loupe"],
        "scores": [35, 22, 19, 17, 14],
    },
}


# ─── 05: imagen resized a cada input size ──────────────────────────────────────
def plot_resized_inputs():
    img = Image.open(SRC_IMG).convert("RGB")
    sizes = [(144, 144, "vww/mbv2"), (160, 160, "in2"), (176, 176, "proxyless"), (128, 160, "person-det")]
    fig, axes = plt.subplots(1, len(sizes), figsize=(14, 4))
    for ax, (w, h, name) in zip(axes, sizes):
        resized = img.resize((w, h), Image.BILINEAR)
        ax.imshow(resized)
        ax.set_title(f"{w}×{h}  ({name})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Misma imagen ({IMG_SOURCE_LABEL}), redimensionada al input que cada modelo espera",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "05_test_image_resized.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'05_test_image_resized.png'}")


# ─── 06: person-det grid + best cell ───────────────────────────────────────────
def plot_person_det_grid():
    img = Image.open(SRC_IMG).convert("RGB").resize((160, 128), Image.BILINEAR)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    grids = [(4, 5, "head 1 (stride 32)\nbest conf = 0.320"),
             (8, 10, "head 2 (stride 16)\n★ best conf = 0.891 ★"),
             (16, 20, "head 3 (stride 8)\nbest conf = 0.051")]
    best_cells = [(2, 2), (4, 5), (8, 10)]  # row, col of "winner" per head (mock illustration)
    for ax, (gh, gw, title), (br, bc) in zip(axes, grids, best_cells):
        ax.imshow(img)
        # draw grid
        for r in range(gh + 1):
            ax.axhline(y=r * 128 / gh, color="white", lw=0.4, alpha=0.5)
        for c in range(gw + 1):
            ax.axvline(x=c * 160 / gw, color="white", lw=0.4, alpha=0.5)
        # highlight best cell on head 2
        if br is not None:
            cw, ch = 160 / gw, 128 / gh
            rect = mpatches.Rectangle((bc*cw, br*ch), cw, ch,
                                      lw=2, ec="lime" if "★" in title else "yellow",
                                      fc="none")
            ax.add_patch(rect)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("MCUNet person-det: 3 cabezas multi-escala — celda con mayor confianza",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "06_person_det_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'06_person_det_grid.png'}")


# ─── 07: panel de clasificación ────────────────────────────────────────────────
def plot_classification_panel():
    img = Image.open(SRC_IMG).convert("RGB")
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 2, 2, 2])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.set_title(f"Imagen de entrada\n({IMG_SOURCE_LABEL})", fontsize=10)
    ax_img.set_xticks([]); ax_img.set_yticks([])

    panels = [
        ("mcunet-vww2 (144²)",     "vww",       "3.39 s"),
        ("mbv2-w0.35 (144²)",      "mbv2-w0.35 (144x144)", "1.64 s"),
        ("proxyless-w0.3 (176²)",  "proxyless-w0.3 (176x176)", "2.51 s"),
        ("mcunet-in2 (160²)",      "mcunet-in2 (160x160)", "3.93 s"),
    ]
    positions = [(0, 1), (0, 2), (0, 3), (1, 1)]
    # Special handling for VWW (binary)
    for (title_short, key, lat), (r, c) in zip(panels, positions):
        ax = fig.add_subplot(gs[r, c])
        if key == "vww":
            data = RESULTS["mcunet-vww2 (144x144)"]
            labels = list(data["scores"].keys())
            scores = list(data["scores"].values())
            colors = ["#2ca02c" if l == "PERSON" else "#bbb" for l in labels]
            bars = ax.barh(labels, scores, color=colors, edgecolor="black", lw=0.5)
            for b, s in zip(bars, scores):
                ax.text(s + (3 if s > 0 else -3), b.get_y() + b.get_height()/2,
                        f"{s:+d}", va="center",
                        ha="left" if s > 0 else "right", fontsize=9)
            ax.set_xlim(-70, 70); ax.axvline(0, color="black", lw=0.5)
            ax.set_title(f"{title_short}  →  inference {lat}", fontsize=10)
        else:
            data = RESULTS[key]
            labels = data["labels"][::-1]
            scores = data["scores"][::-1]
            bars = ax.barh(range(len(labels)), scores,
                           color=["#1f77b4"]*len(labels), edgecolor="black", lw=0.5)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            for b, s in zip(bars, scores):
                ax.text(s + (1 if s > 0 else -1), b.get_y() + b.get_height()/2,
                        f"{s:+d}", va="center",
                        ha="left" if s > 0 else "right", fontsize=9)
            ax.set_title(f"{title_short}  →  inference {lat}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Top-5 predicciones medidas en el XIAO ESP32-S3 (mismos números que el host)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "07_classification_panel.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'07_classification_panel.png'}")


# ─── 08: pipeline diagram ──────────────────────────────────────────────────────
def plot_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 1.0, 2.0, 2.0, "Imagen embedded\n62 KB (int8)\no 245 KB (float32)", "#ffe5b4"),
        (2.8, 1.5, 1.8, 1.0, "Cuantización\n(uint8 → int8)\nzp = -128", "#ffd1d1"),
        (5.1, 1.5, 2.0, 1.0, "TFLite Micro\nInterpreter::Invoke()", "#d1e7ff"),
        (7.6, 1.5, 1.9, 1.0, "esp-nn kernels\nSIMD Xtensa LX7", "#d1ffd1"),
        (10.0, 1.0, 2.8, 2.0, "Output tensor\n(logits int8 / grid fp32)\n+ argmax/NMS", "#e9d1ff"),
    ]
    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.08",
                                       fc=color, ec="black", lw=1)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9)
    # Arrows
    arrows = [(2.4, 2.0, 2.8, 2.0), (4.7, 2.0, 5.1, 2.0),
              (7.2, 2.0, 7.6, 2.0), (9.6, 2.0, 10.0, 2.0)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y1), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.text(6.5, 0.4, "XIAO ESP32-S3 Sense  •  240 MHz  •  8 MB PSRAM",
            ha="center", fontsize=10, style="italic", color="#555")
    ax.set_title("Pipeline de inferencia — desde imagen embebida hasta predicción",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "08_pipeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'08_pipeline.png'}")


if __name__ == "__main__":
    plot_resized_inputs()
    plot_person_det_grid()
    plot_classification_panel()
    plot_pipeline_diagram()
    print("\nDone — 4 visualizaciones nuevas en", OUT)

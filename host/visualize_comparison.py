#!/usr/bin/env python3
"""
Genera un plot comparativo: mismos 4 clasificadores sobre dos imágenes
distintas (placeholder ideal vs frame real de la cámara). Demuestra la
brecha de generalización ("out-of-distribution") de los modelos TinyML
preentrenados.

Salida: docs/plots/09_comparison_ideal_vs_real.png
"""
from __future__ import annotations
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IDEAL = ROOT / "host" / "sample_images" / "person_det.jpg"
REAL  = ROOT / "docs" / "plots" / "sample_camera_frame.png"
OUT   = ROOT / "docs" / "plots" / "09_comparison_ideal_vs_real.png"

# Resultados medidos en HW (idénticos a host)
IDEAL_PREDS = {
    "mcunet-vww2":  ("PERSON (acierta)", "+51"),
    "mcunet-in2":   ("window shade",  "+27"),
    "mbv2-w0.35":   ("bannister",     "-7"),
    "proxyless":    ("obelisk",       "+15"),
    "person-det":   ("PERSON 0.891 (acierta)", "celda 8x10"),
}
REAL_PREDS = {
    "mcunet-vww2":  ("no-pers (falla)", "+22"),
    "mcunet-in2":   ("alp",           "+27"),
    "mbv2-w0.35":   ("jersey",        "-3"),
    "proxyless":    ("power drill",   "+35"),
    "person-det":   ("no-pers 0.419 (falla)", "umbral 0.5"),
}


def main():
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.18)

    # Row 1: ideal image + table
    ax_img1 = fig.add_subplot(gs[0, 0])
    ax_img1.imshow(Image.open(IDEAL))
    ax_img1.set_title("Imagen del repo MCUNet (person_det.jpg)", fontsize=11)
    ax_img1.set_xticks([]); ax_img1.set_yticks([])

    ax_tab1 = fig.add_subplot(gs[0, 1])
    ax_tab1.axis("off")
    cell_data = [[k, v[0], v[1]] for k, v in IDEAL_PREDS.items()]
    tbl = ax_tab1.table(
        cellText=cell_data,
        colLabels=["Modelo", "Top-1 / Predicción", "Score"],
        colWidths=[0.32, 0.45, 0.20],
        cellLoc="left", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for i in range(1, len(cell_data) + 1):
        if "acierta" in cell_data[i-1][1]:
            tbl[(i, 1)].set_facecolor("#d6ffd6")
    ax_tab1.set_title("Predicciones — imagen ideal\n(personas centradas, fondo neutro)",
                      fontsize=11, pad=10)

    # Row 2: real image + table
    ax_img2 = fig.add_subplot(gs[1, 0])
    ax_img2.imshow(Image.open(REAL))
    ax_img2.set_title("Imagen capturada por la OV2640 del Sense\n(techo + sujeto desde abajo)",
                      fontsize=11)
    ax_img2.set_xticks([]); ax_img2.set_yticks([])

    ax_tab2 = fig.add_subplot(gs[1, 1])
    ax_tab2.axis("off")
    cell_data = [[k, v[0], v[1]] for k, v in REAL_PREDS.items()]
    tbl = ax_tab2.table(
        cellText=cell_data,
        colLabels=["Modelo", "Top-1 / Predicción", "Score"],
        colWidths=[0.32, 0.45, 0.20],
        cellLoc="left", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for i in range(1, len(cell_data) + 1):
        if "falla" in cell_data[i-1][1]:
            tbl[(i, 1)].set_facecolor("#ffe0d6")
    ax_tab2.set_title("Predicciones — imagen capturada\n(ángulo y composición fuera del training set)",
                      fontsize=11, pad=10)

    fig.suptitle(
        "Predicciones sobre dos imágenes — modelos TinyML preentrenados sin fine-tuning",
        fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""
Evalúa TFLite (fp32 y int8) sobre el test set de ShanghaiTech Part B
y compara con la referencia PyTorch.

Imprime una tabla:
    Model         MAE       RMSE      mean_pred  mean_true
    -----------   -------   -------   ---------  ---------
    PT (CSV ref)  16.835    25.784    121.788    123.801
    TFLite fp32   ...       ...       ...        ...
    TFLite int8   ...       ...       ...        ...

Uso:
    python eval_tflite.py
    python eval_tflite.py --limit 50
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image

import tensorflow as tf


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"),
                      dtype=np.float32) / 255.0


def gt_count(npy_path: Path) -> float:
    """Cuenta de puntos anotados, equivale a sum(density)."""
    pts = np.load(npy_path)
    return float(len(pts))


def run_tflite(model_path: Path, images_dir: Path, ann_dir: Path,
               ids: list[str], num_threads: int = 8,
               post_softplus: bool = False
               ) -> tuple[list[float], list[float]]:
    """Detecta tipo de input/output del modelo automáticamente:
    si dtype != float32, asume cuantizado y aplica scale/zp."""
    inter = tf.lite.Interpreter(model_path=str(model_path),
                                num_threads=num_threads)
    inter.allocate_tensors()
    ind = inter.get_input_details()[0]
    outd = inter.get_output_details()[0]
    in_int = ind["dtype"] != np.float32
    out_int = outd["dtype"] != np.float32
    if in_int:
        in_scale = float(ind["quantization_parameters"]["scales"][0])
        in_zp = int(ind["quantization_parameters"]["zero_points"][0])
    if out_int:
        out_scale = float(outd["quantization_parameters"]["scales"][0])
        out_zp = int(outd["quantization_parameters"]["zero_points"][0])

    preds, trues = [], []
    for i, sid in enumerate(ids):
        img = load_image(images_dir / f"{sid}.jpg")[None, ...]
        if in_int:
            x = np.round(img / in_scale + in_zp).clip(-128, 127).astype(ind["dtype"])
        else:
            x = img.astype(np.float32)
        inter.set_tensor(ind["index"], x)
        inter.invoke()
        y = inter.get_tensor(outd["index"])
        if out_int:
            y_f = (y.astype(np.float32) - out_zp) * out_scale
        else:
            y_f = y
        if post_softplus:
            # softplus numéricamente estable
            y_f = np.where(y_f > 20, y_f, np.log1p(np.exp(np.minimum(y_f, 20))))
        preds.append(float(y_f.sum()))
        trues.append(gt_count(ann_dir / f"{sid}.npy"))
        if (i + 1) % 25 == 0 or i == len(ids) - 1:
            print(f"    {i+1}/{len(ids)}", flush=True)
    return preds, trues


def mae_rmse(preds, trues):
    a = np.asarray(preds, dtype=np.float64)
    b = np.asarray(trues, dtype=np.float64)
    return float(np.mean(np.abs(a - b))), float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="~/work/geomcu-deploy")
    p.add_argument("--limit", type=int, default=0,
                   help="evaluar solo las primeras N imgs (0=todas)")
    p.add_argument("--threads", type=int, default=8,
                   help="num_threads del TFLite Interpreter")
    args = p.parse_args()

    ws = Path(os.path.expanduser(args.workspace))
    images_dir = ws / "dataset" / "part_B" / "test_data" / "images"
    ann_dir = ws / "dataset" / "part_B" / "test_data" / "annotations"
    csv_ref = ws / "run" / "detailed_eval" / "per_image_predictions.csv"

    ids = sorted(f.stem for f in images_dir.glob("*.jpg"))
    if args.limit:
        ids = ids[: args.limit]
    print(f"evaluando {len(ids)} imgs test\n")

    # Referencia del CSV del repo (corrió en GPU, fp32)
    with csv_ref.open() as f:
        ref = {row["id"]: float(row["pred_count"])
               for row in csv.DictReader(f)}
    ref_preds = [ref[s] for s in ids]
    ref_trues = [gt_count(ann_dir / f"{s}.npy") for s in ids]
    ref_mae, ref_rmse = mae_rmse(ref_preds, ref_trues)

    variants = [
        ("TFLite fp32",            "microwd_paper_fp32.tflite", False),
        ("TFLite int8 (I/O)",      "microwd_paper_int8.tflite", False),
        ("TFLite int8 + fp32 out", "microwd_paper_int8_fp32out.tflite", False),
        ("TFLite dyn-range",       "microwd_paper_dynamic.tflite", False),
        ("TFLite logits int8 + fp32 softplus",
                                   "microwd_paper_logits_int8.tflite", True),
    ]
    results = []
    for label, fname, post_sp in variants:
        path = ws / "tflite" / fname
        if not path.exists():
            print(f"\n[skip] {label}: {path.name} no existe")
            continue
        print(f"\n== {label} ==")
        preds, trues = run_tflite(path, images_dir, ann_dir, ids,
                                  num_threads=args.threads,
                                  post_softplus=post_sp)
        mae, rmse = mae_rmse(preds, trues)
        results.append((label, mae, rmse, np.mean(preds), np.mean(trues)))

    print()
    print(f"{'Model':<26} {'MAE':>8} {'RMSE':>8} {'mean_pred':>11} {'mean_true':>11}")
    print("-" * 68)
    print(f"{'PT (CSV ref)':<26} {ref_mae:>8.3f} {ref_rmse:>8.3f} "
          f"{np.mean(ref_preds):>11.3f} {np.mean(ref_trues):>11.3f}")
    for label, mae, rmse, mp, mt in results:
        print(f"{label:<26} {mae:>8.3f} {rmse:>8.3f} {mp:>11.3f} {mt:>11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Keras model (microwd_paper.keras con pesos del best_mae.pt) -> TFLite.

Produce 2 archivos:
  - microwd_paper_fp32.tflite : float32, para validar la conversión sin pérdida
  - microwd_paper_int8.tflite : full-integer int8 (input/output int8), para ESP32

Representative dataset = N imágenes train de ShanghaiTech Part B.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image

import tensorflow as tf


def build_representative_dataset(images_dir: Path, n: int = 100, seed: int = 0,
                                 target_hw: tuple[int, int] | None = None):
    """Si target_hw está dado, recorta cada imagen a esa resolución
    (random crop) para alimentar al modelo cuantizado con el mismo input
    shape que el modelo Keras."""
    files = sorted(images_dir.glob("*.jpg"))
    random.Random(seed).shuffle(files)
    files = files[:n]
    print(f"  representative dataset: {len(files)} imgs from {images_dir.name}")
    if target_hw is not None:
        print(f"    random-crop to {target_hw[0]}x{target_hw[1]}")
    rng = random.Random(seed + 1)

    def gen():
        for f in files:
            img = np.asarray(Image.open(f).convert("RGB"),
                             dtype=np.float32) / 255.0
            if target_hw is not None:
                th, tw = target_hw
                H, W, _ = img.shape
                y0 = rng.randint(0, max(0, H - th))
                x0 = rng.randint(0, max(0, W - tw))
                img = img[y0:y0+th, x0:x0+tw]
            yield [img[None, ...]]
    return gen


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="~/work/geomcu-deploy")
    p.add_argument("--n-rep", type=int, default=100,
                   help="imgs en representative dataset")
    p.add_argument("--only", default=None,
                   choices=["fp32", "int8", "int8_fp32out", "dynamic",
                            "logits_int8"],
                   help="convertir sólo un variant (skip otros)")
    p.add_argument("--input-shape", default="768,1024",
                   help="H,W del input del modelo Keras. Para deploy patch-based "
                   "se cuantiza con la shape del patch (ej. 256,320).")
    p.add_argument("--suffix", default="",
                   help="sufijo opcional para nombres de salida (ej. _patch)")
    args = p.parse_args()

    ws = Path(os.path.expanduser(args.workspace))
    train_dir = ws / "dataset" / "part_B" / "train_data" / "images"
    suf = args.suffix
    out_fp32 = ws / "tflite" / f"microwd_paper_fp32{suf}.tflite"
    out_int8 = ws / "tflite" / f"microwd_paper_int8{suf}.tflite"
    out_int8_fp32out = ws / "tflite" / f"microwd_paper_int8_fp32out{suf}.tflite"
    out_dynr = ws / "tflite" / f"microwd_paper_dynamic{suf}.tflite"
    out_logits_int8 = ws / "tflite" / f"microwd_paper_logits_int8{suf}.tflite"
    out_fp32.parent.mkdir(parents=True, exist_ok=True)
    H, W = (int(x) for x in args.input_shape.split(","))

    # Construir modelo Keras y cargar pesos desde best_mae.pt en memoria.
    # Saltamos el save/load del .keras porque h5py 3.x + tf 2.15 dan problemas.
    print("building Keras model and loading PT weights ...")
    from port_weights import _keras_to_pt_conv, _keras_to_pt_dw, _keras_to_pt_bn_base
    from model_microwd_keras import build_microwd
    import torch
    # El variant "logits_int8" usa el modelo SIN el softplus final para que la
    # int8 quantization capture mejor el rango de los logits. Softplus se
    # aplica en fp32 post-proceso.
    drop_softplus = args.only == "logits_int8"
    model = build_microwd(input_shape=(H, W, 3),
                          drop_final_activation=drop_softplus)
    ckpt = ws / "run" / "best_mae.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    for layer in model.layers:
        if not layer.weights:
            continue
        name = layer.name
        if name.endswith("_conv") and not name.endswith("_dw_conv"):
            k = _keras_to_pt_conv(name)
            w = state[k].numpy().transpose(2, 3, 1, 0)
            if layer.use_bias:
                b = state[k.replace("weight", "bias")].numpy()
                layer.set_weights([w, b])
            else:
                layer.set_weights([w])
        elif name.endswith("_dw"):
            k = _keras_to_pt_dw(name)
            w = state[k].numpy().transpose(2, 3, 0, 1)
            layer.set_weights([w])
        elif name.endswith("_bn"):
            b = _keras_to_pt_bn_base(name)
            layer.set_weights([state[f"{b}.weight"].numpy(),
                               state[f"{b}.bias"].numpy(),
                               state[f"{b}.running_mean"].numpy(),
                               state[f"{b}.running_var"].numpy()])
        elif name == "out":
            w = state["out.weight"].numpy().transpose(2, 3, 1, 0)
            layer.set_weights([w, state["out.bias"].numpy()])
    model.trainable = False
    print(f"  params: {model.count_params():,}")

    # --- TFLite float32 ---
    if args.only in (None, "fp32"):
        print("\n== TFLite float32 ==")
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_fp32 = conv.convert()
        out_fp32.write_bytes(tflite_fp32)
        print(f"  wrote {out_fp32} ({len(tflite_fp32)/1024:.1f} KB)")

    # --- TFLite full integer int8 ---
    if args.only in (None, "int8"):
        print("\n== TFLite int8 (full integer quantization) ==")
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = build_representative_dataset(
            train_dir, args.n_rep, target_hw=(H, W))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        tflite_int8 = conv.convert()
        out_int8.write_bytes(tflite_int8)
        print(f"  wrote {out_int8} ({len(tflite_int8)/1024:.1f} KB)")

    # --- TFLite int8 con output FP32 (output denso es muy chico vs scale int8) ---
    if args.only in (None, "int8_fp32out"):
        print("\n== TFLite int8 con output FP32 ==")
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = build_representative_dataset(
            train_dir, args.n_rep, target_hw=(H, W))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.float32
        tflite_mixed = conv.convert()
        out_int8_fp32out.write_bytes(tflite_mixed)
        print(f"  wrote {out_int8_fp32out} ({len(tflite_mixed)/1024:.1f} KB)")

    # --- TFLite dynamic range (weights int8, activaciones fp32) ---
    if args.only in (None, "dynamic"):
        print("\n== TFLite dynamic range (weights int8) ==")
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_dyn = conv.convert()
        out_dynr.write_bytes(tflite_dyn)
        print(f"  wrote {out_dynr} ({len(tflite_dyn)/1024:.1f} KB)")

    # --- Logits int8 (sin softplus). El modelo aquí YA está construido
    # sin softplus (drop_final_activation=True via --only logits_int8). ---
    if args.only == "logits_int8":
        print("\n== TFLite int8 SIN softplus (logits crudos) ==")
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = build_representative_dataset(
            train_dir, args.n_rep, target_hw=(H, W))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        tflite_logits = conv.convert()
        out_logits_int8.write_bytes(tflite_logits)
        print(f"  wrote {out_logits_int8} ({len(tflite_logits)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

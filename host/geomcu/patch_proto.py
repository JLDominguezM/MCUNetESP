"""
Prototipo en host de inferencia patch-based (MCUNetV2 style) sobre
el .tflite full-resolution.

Algoritmo:
  - Imagen 768x1024 -> tile en grid n_h × n_w patches del mismo tamaño.
  - Cada patch se expande con halo (replicate padding) para preservar
    el campo receptivo. Patch interior + halo = input al modelo.
  - Se crea un TFLite Interpreter con input resized al tamaño del patch
    aumentado, y se invoca por patch.
  - De cada output se recorta la zona "interior" (sin halo) y se pega
    en el mapa de densidad final stride-4.
  - Se compara contra correr el modelo sobre la imagen completa.

Objetivo: validar el harness con un .tflite cualquiera (fp32 sirve)
antes de invertir tiempo en C++ en el ESP32.

Uso:
    python patch_proto.py
    python patch_proto.py --tflite microwd_paper_int8_fp32out.tflite --tiles 2 2 --halo 64
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image

import tensorflow as tf


def load_image_float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"),
                      dtype=np.float32) / 255.0


def _softplus(x: np.ndarray) -> np.ndarray:
    """softplus numéricamente estable."""
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def run_full(tflite_path: Path, img: np.ndarray, num_threads: int = 16,
             post_softplus: bool = False) -> np.ndarray:
    inter = tf.lite.Interpreter(model_path=str(tflite_path),
                                num_threads=num_threads)
    inter.allocate_tensors()
    ind = inter.get_input_details()[0]
    outd = inter.get_output_details()[0]
    x = _quantize_input(img[None], ind)
    inter.set_tensor(ind["index"], x)
    inter.invoke()
    y = _dequantize_output(inter.get_tensor(outd["index"]), outd)[0, ..., 0]
    return _softplus(y) if post_softplus else y


def run_patched(tflite_path: Path, img: np.ndarray, tiles: tuple[int, int],
                halo: int, stride: int = 4, num_threads: int = 16,
                post_softplus: bool = False) -> np.ndarray:
    """Tile la imagen en tiles[0] x tiles[1] patches del mismo tamaño con halo."""
    H, W, _ = img.shape
    n_h, n_w = tiles
    assert H % (n_h * stride) == 0 and W % (n_w * stride) == 0, \
        "tamaño imagen debe ser múltiplo de tiles*stride"
    patch_h, patch_w = H // n_h, W // n_w  # input interior (sin halo)

    # patch con halo
    in_h = patch_h + 2 * halo
    in_w = patch_w + 2 * halo
    print(f"  patch interior: {patch_h}x{patch_w}, con halo {halo}: input {in_h}x{in_w}")

    # Padea la imagen para que patches en la borde tengan halo "válido".
    # mode='edge' replica el píxel borde, empíricamente mejor que constant=0
    # porque preserva la estadística natural de la imagen y el modelo no
    # entrenó con bordes "tipo box". Mantenemos eso aquí.
    padded = np.pad(img, ((halo, halo), (halo, halo), (0, 0)), mode="edge")

    inter = tf.lite.Interpreter(model_path=str(tflite_path),
                                num_threads=num_threads)
    ind0 = inter.get_input_details()[0]
    print(f"  TFLite original input shape: {ind0['shape']}")
    inter.resize_tensor_input(ind0["index"], [1, in_h, in_w, 3])
    inter.allocate_tensors()
    ind = inter.get_input_details()[0]
    outd = inter.get_output_details()[0]
    print(f"  TFLite resized input shape : {ind['shape']}")
    print(f"  TFLite output shape        : {outd['shape']}")

    # Output completo stride-4
    out_full = np.zeros((H // stride, W // stride), dtype=np.float32)

    for i in range(n_h):
        for j in range(n_w):
            y0 = i * patch_h
            x0 = j * patch_w
            # En coords de la imagen padeada, el patch+halo empieza en (y0, x0)
            chunk = padded[y0:y0 + in_h, x0:x0 + in_w][None]  # (1, in_h, in_w, 3)
            x = _quantize_input(chunk, ind)
            inter.set_tensor(ind["index"], x)
            inter.invoke()
            y = _dequantize_output(inter.get_tensor(outd["index"]), outd)
            if post_softplus:
                y = _softplus(y)
            # y shape (1, in_h/4, in_w/4, 1), extraer interior
            halo_s = halo // stride
            interior_h = patch_h // stride
            interior_w = patch_w // stride
            interior = y[0,
                         halo_s:halo_s + interior_h,
                         halo_s:halo_s + interior_w, 0]
            out_full[i * interior_h:(i + 1) * interior_h,
                     j * interior_w:(j + 1) * interior_w] = interior
            print(f"    patch ({i},{j}) -> count {interior.sum():.3f}", flush=True)
    return out_full


def _quantize_input(x: np.ndarray, det) -> np.ndarray:
    if det["dtype"] == np.float32:
        return x.astype(np.float32)
    scale = float(det["quantization_parameters"]["scales"][0])
    zp = int(det["quantization_parameters"]["zero_points"][0])
    return np.round(x / scale + zp).clip(-128, 127).astype(det["dtype"])


def _dequantize_output(y: np.ndarray, det) -> np.ndarray:
    if det["dtype"] == np.float32:
        return y.astype(np.float32)
    scale = float(det["quantization_parameters"]["scales"][0])
    zp = int(det["quantization_parameters"]["zero_points"][0])
    return (y.astype(np.float32) - zp) * scale


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="~/work/geomcu-deploy")
    p.add_argument("--tflite", default="microwd_paper_fp32.tflite")
    p.add_argument("--image", default="IMG_1")
    p.add_argument("--tiles", nargs=2, type=int, default=[2, 2],
                   help="filas columnas de patches")
    p.add_argument("--halo", type=int, default=64,
                   help="pixeles de halo en cada lado del patch")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--post-softplus", action="store_true",
                   help="aplicar softplus al dequant (para variant logits_int8)")
    args = p.parse_args()

    ws = Path(os.path.expanduser(args.workspace))
    tflite_path = ws / "tflite" / args.tflite
    img_path = ws / "dataset" / "part_B" / "test_data" / "images" / f"{args.image}.jpg"
    print(f"tflite : {tflite_path}")
    print(f"image  : {img_path}")
    print(f"tiles  : {args.tiles}, halo {args.halo}\n")

    img = load_image_float(img_path)
    print(f"image shape: {img.shape}\n")

    print("== full-image forward ==")
    d_full = run_full(tflite_path, img, num_threads=args.threads,
                      post_softplus=args.post_softplus)
    count_full = float(d_full.sum())
    print(f"  count_full = {count_full:.4f}\n")

    print(f"== patched forward ({args.tiles[0]}x{args.tiles[1]}, halo {args.halo}) ==")
    d_patched = run_patched(tflite_path, img, tuple(args.tiles), args.halo,
                            num_threads=args.threads,
                            post_softplus=args.post_softplus)
    count_patched = float(d_patched.sum())
    print(f"  count_patched = {count_patched:.4f}")
    print(f"  delta = {count_patched - count_full:+.4f}  "
          f"({(count_patched - count_full) / max(count_full, 1e-6) * 100:+.2f}%)")
    print(f"  density max diff = {np.abs(d_patched - d_full).max():.6f}")
    print(f"  density mean diff = {np.abs(d_patched - d_full).mean():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

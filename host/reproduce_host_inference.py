#!/usr/bin/env python3
"""
Reproducibility check: corre los 5 modelos sobre la imagen de test en el host
e imprime los mismos top-5 que reporta el XIAO. Si los números aquí coinciden
con los logs de `docs/logs/*.log`, el deploy en ESP32 es bit-correcto.

Uso:
    conda activate mcunet
    python host/reproduce_host_inference.py
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
from PIL import Image

# silenciar tensorflow noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.lite.python.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent
IMG  = ROOT / "host" / "sample_images" / "person_det.jpg"

# (alias, .tflite filename, expected_input_WH)
MODELS = [
    ("mcunet-vww2",       "mcunet-vww2.tflite",       144),
    ("mcunet-in2",        "mcunet-in2.tflite",        160),
    ("mbv2-w0.35",        "mbv2-w0.35.tflite",        144),
    ("proxyless-w0.3",    "proxyless-w0.3.tflite",    176),
    # person-det es float32 + 3 outputs grid → mejor verificarlo aparte
]


def quantize_int8(img_arr_uint8: np.ndarray) -> np.ndarray:
    """RGB uint8 → int8 con zero_point = -128 (esquema simétrico estándar)."""
    return (img_arr_uint8.astype(np.int16) - 128).astype(np.int8)


def run(alias: str, fname: str, sz: int) -> None:
    path = ROOT / "models" / fname
    if not path.exists():
        print(f"  SKIP — {path} no existe. Corre `python host/download_models.py` primero.")
        return
    it = Interpreter(model_path=str(path)); it.allocate_tensors()
    ind = it.get_input_details()[0]; outd = it.get_output_details()[0]
    img = Image.open(IMG).convert("RGB").resize((sz, sz), Image.BILINEAR)
    arr = quantize_int8(np.asarray(img))[None, ...]
    it.set_tensor(ind["index"], arr); it.invoke()
    out = it.get_tensor(outd["index"])[0]

    if out.size == 2:  # VWW: 2 logits
        s = [int(out[0]), int(out[1])]
        verdict = "PERSON" if s[1] > s[0] else "no-pers"
        print(f"  {alias:20s}  input {sz}x{sz}  → scores=[{s[0]:+d}, {s[1]:+d}]  → {verdict}")
    else:
        top5 = np.argsort(out)[-5:][::-1].tolist()
        scores = [int(out[i]) for i in top5]
        print(f"  {alias:20s}  input {sz}x{sz}  → top5={top5}  scores={scores}")


def main() -> int:
    if not IMG.exists():
        print(f"ERROR: imagen de test no existe: {IMG}", file=sys.stderr)
        return 1
    print(f"Reproducibility check (host TFLite vs ESP32 logs en docs/logs/)")
    print(f"  imagen: {IMG.name}\n")
    for alias, fname, sz in MODELS:
        run(alias, fname, sz)
    print("\nCompara estos números contra `docs/logs/*.log` — deben coincidir bit-a-bit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

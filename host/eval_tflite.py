#!/usr/bin/env python3
"""
Valida un modelo .tflite en host antes de flashearlo al ESP32-S3.

Uso:
    python eval_tflite.py ../models/mcunet-vww2.tflite ./sample_images/*.jpg

Imprime:
  - shapes de input/output
  - dtype (int8 esperado)
  - top-1 / top-5 por imagen
  - tiempo medio de inferencia (CPU host, solo de referencia)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


def load_image(path: Path, size: tuple[int, int], channels: int) -> np.ndarray:
    img = Image.open(path)
    img = img.convert("L" if channels == 1 else "RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    if channels == 1:
        arr = arr[..., None]
    return arr


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("uso: eval_tflite.py <model.tflite> <img1> [img2 ...]")
        return 2

    model_path = Path(argv[1])
    img_paths = [Path(p) for p in argv[2:]]

    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    print(f"Model:  {model_path.name}")
    print(f"Input:  {in_det['shape']}  dtype={in_det['dtype']}")
    print(f"Output: {out_det['shape']}  dtype={out_det['dtype']}")

    _, h, w, c = in_det["shape"]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    latencies: list[float] = []
    for p in img_paths:
        img = load_image(p, (w, h), c).astype(np.float32) / 255.0
        if in_det["dtype"] == np.int8:
            q = np.round(img / in_scale + in_zp).astype(np.int8)
        else:
            q = img.astype(in_det["dtype"])
        q = q[None, ...]

        interp.set_tensor(in_det["index"], q)
        t0 = time.perf_counter()
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        out = interp.get_tensor(out_det["index"])[0]
        if out.dtype == np.int8:
            out = (out.astype(np.float32) - out_zp) * out_scale
        top5 = np.argsort(out)[-5:][::-1]
        print(f"  {p.name}: top5={top5.tolist()}  scores={[round(float(out[i]), 3) for i in top5]}")

    if latencies:
        print(f"Avg host latency: {np.mean(latencies):.1f} ms over {len(latencies)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

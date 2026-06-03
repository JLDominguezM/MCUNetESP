"""
Lista las operaciones únicas usadas por cada .tflite, su número de instancias,
y los rangos de quantización. Sirve para planear el MicroMutableOpResolver
del firmware ESP32 y detectar ops no soportados por esp-tflite-micro.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


def analyze(path: Path) -> None:
    print(f"\n=== {path.name} ({path.stat().st_size/1024:.1f} KB) ===")
    inter = tf.lite.Interpreter(model_path=str(path))
    inter.allocate_tensors()
    ind = inter.get_input_details()[0]
    outd = inter.get_output_details()[0]
    print(f"  input  dtype={ind['dtype'].__name__:10} shape={tuple(ind['shape'])}")
    print(f"  output dtype={outd['dtype'].__name__:10} shape={tuple(outd['shape'])}")

    ops = Counter()
    for op in inter._get_ops_details():
        ops[op["op_name"]] += 1
    print(f"  ops ({sum(ops.values())} total, {len(ops)} unique):")
    for name, cnt in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"    {name:30} {cnt:3}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="~/work/geomcu-deploy")
    args = p.parse_args()
    ws = Path(os.path.expanduser(args.workspace))
    tdir = ws / "tflite"
    for f in sorted(tdir.glob("*.tflite")):
        analyze(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

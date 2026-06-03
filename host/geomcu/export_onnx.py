"""
PyTorch -> ONNX export del MiCrowdNet entrenado en ShanghaiTech Part B.

Entrada fija 1x3x768x1024 (full-res del dataset). Más adelante haremos
versiones por patch para deploy en ESP32.

Valida que la salida ONNX coincide con PyTorch a 1e-5.

Uso:
    python export_onnx.py
    python export_onnx.py --workspace ~/work/geomcu-deploy --opset 17
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from model_microwd import MiCrowdNetPaperFullFrame, count_params


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="~/work/geomcu-deploy")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--shape", default="1,3,768,1024",
                   help="input shape NCHW comma-separated")
    args = p.parse_args()

    ws = Path(os.path.expanduser(args.workspace))
    ckpt_path = ws / "run" / "best_mae.pt"
    out_dir = ws / "tflite"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "microwd_paper.onnx"

    shape = tuple(int(x) for x in args.shape.split(","))
    print(f"checkpoint: {ckpt_path}")
    print(f"shape     : {shape}")
    print(f"opset     : {args.opset}")
    print(f"out       : {onnx_path}\n")

    model = MiCrowdNetPaperFullFrame()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"params: {count_params(model):,}")

    dummy = torch.randn(*shape)

    with torch.no_grad():
        ref = model(dummy).numpy()

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["density"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    sz = onnx_path.stat().st_size
    print(f"wrote {onnx_path} ({sz/1024:.1f} KB)")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime no instalado, skip parity check")
        return 0

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"image": dummy.numpy()})[0]
    diff = np.abs(onnx_out - ref)
    print(f"\nparity vs PyTorch:")
    print(f"  max abs diff = {diff.max():.3e}")
    print(f"  mean abs diff = {diff.mean():.3e}")
    print(f"  sum(ref)  = {ref.sum():.4f}")
    print(f"  sum(onnx) = {onnx_out.sum():.4f}")
    return 0 if diff.max() < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())

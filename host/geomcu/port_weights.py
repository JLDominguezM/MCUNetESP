"""
Copia pesos del state_dict PyTorch (best_mae.pt) al modelo Keras
(model_microwd_keras.py) y guarda el modelo Keras con pesos
ya cargados.

Verificación: tras cargar, corre forward sobre la misma imagen en
PyTorch y Keras y compara el conteo.

Uso:
    python port_weights.py
    python port_weights.py --workspace ~/work/geomcu-deploy --image IMG_1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only; full-res activations OOM en GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch

import tensorflow as tf
from PIL import Image

from model_microwd import MiCrowdNetPaperFullFrame
from model_microwd_keras import build_microwd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="~/work/geomcu-deploy")
    p.add_argument("--image", default="IMG_1", help="id imagen test para parity")
    args = p.parse_args()

    ws = Path(os.path.expanduser(args.workspace))
    ckpt = ws / "run" / "best_mae.pt"
    img_path = ws / "dataset" / "part_B" / "test_data" / "images" / f"{args.image}.jpg"
    out_keras = ws / "tflite" / "microwd_paper.keras"
    out_keras.parent.mkdir(parents=True, exist_ok=True)

    # --- PyTorch ---
    print(f"loading {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    pt = MiCrowdNetPaperFullFrame()
    pt.load_state_dict(state, strict=True)
    pt.eval()
    print(f"  PT params: {sum(p.numel() for p in pt.parameters()):,}")

    # --- Keras ---
    keras_model = build_microwd(input_shape=(768, 1024, 3))
    print(f"  Keras trainable: {sum(v.numpy().size for v in keras_model.trainable_variables):,}")

    # --- Mapeo ---
    # Para cada layer Keras con pesos, buscar la(s) entry(ies) en state_dict y asignar.
    n_assigned = 0
    n_layers_touched = 0
    for layer in keras_model.layers:
        w = layer.weights
        if not w:
            continue
        name = layer.name

        # Conv2D estándar (no DW)
        if name.endswith("_conv") and not name.endswith("_dw_conv"):
            pt_key = _keras_to_pt_conv(name)
            w_pt = state[pt_key].numpy()  # (out, in, kH, kW)
            w_tf = np.transpose(w_pt, (2, 3, 1, 0))
            if layer.use_bias:
                # algunos conv tienen bias (out), sólo "out"
                bias_key = pt_key.replace("weight", "bias")
                b_pt = state[bias_key].numpy()
                layer.set_weights([w_tf, b_pt])
            else:
                layer.set_weights([w_tf])
            n_assigned += sum(t.size for t in layer.get_weights())
            n_layers_touched += 1

        elif name.endswith("_dw"):  # DepthwiseConv2D
            pt_key = _keras_to_pt_dw(name)
            w_pt = state[pt_key].numpy()  # (in, 1, kH, kW)
            w_tf = np.transpose(w_pt, (2, 3, 0, 1))  # (kH, kW, in, 1)
            layer.set_weights([w_tf])
            n_assigned += w_tf.size
            n_layers_touched += 1

        elif name.endswith("_bn"):
            base = _keras_to_pt_bn_base(name)
            gamma = state[f"{base}.weight"].numpy()
            beta = state[f"{base}.bias"].numpy()
            mean = state[f"{base}.running_mean"].numpy()
            var = state[f"{base}.running_var"].numpy()
            layer.set_weights([gamma, beta, mean, var])
            n_assigned += gamma.size + beta.size + mean.size + var.size
            n_layers_touched += 1

        elif name == "out":
            w_pt = state["out.weight"].numpy()  # (1, 30, 1, 1)
            b_pt = state["out.bias"].numpy()  # (1,)
            w_tf = np.transpose(w_pt, (2, 3, 1, 0))  # (1, 1, 30, 1)
            layer.set_weights([w_tf, b_pt])
            n_assigned += w_tf.size + b_pt.size
            n_layers_touched += 1

    print(f"  layers touched: {n_layers_touched}, params assigned: {n_assigned:,}")

    # --- Parity check ---
    img = np.asarray(Image.open(img_path).convert("RGB"),
                     dtype=np.float32) / 255.0
    x_nhwc = img[None, ...]
    x_nchw = torch.from_numpy(np.transpose(x_nhwc, (0, 3, 1, 2)))

    with torch.no_grad():
        d_pt = pt(x_nchw).numpy()  # (1, 1, 192, 256)
    d_keras = keras_model(x_nhwc, training=False).numpy()  # (1, 192, 256, 1)
    d_keras_nchw = np.transpose(d_keras, (0, 3, 1, 2))

    diff = np.abs(d_keras_nchw - d_pt)
    print()
    print(f"parity ({args.image}):")
    print(f"  PT    count = {d_pt.sum():.4f}")
    print(f"  Keras count = {d_keras.sum():.4f}")
    print(f"  max abs diff  = {diff.max():.3e}")
    print(f"  mean abs diff = {diff.mean():.3e}")

    if diff.max() < 1e-3:
        keras_model.save(str(out_keras))
        print(f"\nsaved {out_keras}")
        return 0
    else:
        print("\nFAIL: mismatch > 1e-3, no se guarda el modelo")
        return 1


def _keras_to_pt_conv(name: str) -> str:
    # branch1_b1_expand_conv -> branch1.block1.expand.conv.weight
    # branch1_b1_project_conv -> branch1.block1.project.0.weight
    parts = name.split("_")
    branch = parts[0]
    block_n = parts[1][1:]  # "b1" -> "1"
    kind = "_".join(parts[2:-1])  # expand / project
    if kind == "project":
        return f"{branch}.block{block_n}.project.0.weight"
    return f"{branch}.block{block_n}.{kind}.conv.weight"


def _keras_to_pt_dw(name: str) -> str:
    # branch1_b1_dw_dw -> branch1.block1.depthwise.conv.weight
    parts = name.split("_")
    branch = parts[0]
    block_n = parts[1][1:]
    return f"{branch}.block{block_n}.depthwise.conv.weight"


def _keras_to_pt_bn_base(name: str) -> str:
    # branch1_b1_expand_bn -> branch1.block1.expand.bn
    # branch1_b1_dw_bn -> branch1.block1.depthwise.bn
    # branch1_b1_project_bn -> branch1.block1.project.1   <-- BN es 2do elemento del Sequential
    parts = name.split("_")
    branch = parts[0]
    block_n = parts[1][1:]
    kind = "_".join(parts[2:-1])  # expand / dw / project
    if kind == "project":
        return f"{branch}.block{block_n}.project.1"
    if kind == "dw":
        return f"{branch}.block{block_n}.depthwise.bn"
    return f"{branch}.block{block_n}.{kind}.bn"


if __name__ == "__main__":
    raise SystemExit(main())

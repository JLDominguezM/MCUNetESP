"""
Port Keras de MiCrowdNetPaperFullFrame.

Topología 1:1 con model_microwd.py (PyTorch). Diseñado para cargar los pesos
del .pt entrenado y exportar a TFLite con TFLiteConverter.from_keras_model
sin pasar por ONNX.

Notas:
- Tensorflow es NHWC; PyTorch es NCHW. Las shapes que devuelven los layers
  son distintas, pero los pesos sólo necesitan transpose al cargar.
- SamePadDepthwiseConv tiene padding asimétrico explícito para kernels pares
  (16 → pl=7, pr=8). Lo replicamos con tf.pad antes del DepthwiseConv2D
  (padding='valid').
- ReLU(inplace=True) en PyTorch no afecta numerics, equivale a ReLU sin más.
- nn.BatchNorm2d default eps=1e-5 (=Keras BN default). momentum no aplica
  en inferencia.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _conv_bn_relu(x, filters, k=3, s=1, groups=1, name=None):
    """Equiv a ConvBNReLU(cin, cout, k, s, p=k//2, groups). bias=False."""
    if isinstance(k, int):
        pad_h = pad_w = k // 2
    else:
        pad_h, pad_w = k[0] // 2, k[1] // 2
    if pad_h or pad_w:
        x = layers.ZeroPadding2D(((pad_h, pad_h), (pad_w, pad_w)),
                                 name=f"{name}_pad")(x)
    if groups == 1:
        x = layers.Conv2D(filters, k, strides=s, padding="valid",
                          use_bias=False, name=f"{name}_conv")(x)
    else:
        x = layers.DepthwiseConv2D(k, strides=s, padding="valid",
                                   use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(epsilon=1e-5, name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def _same_pad_dw(x, kernel_size, name):
    """SamePadDepthwiseConv: F.pad asimétrico + DW conv + BN + ReLU."""
    k = kernel_size
    pad_total = k - 1
    pl, pr = pad_total // 2, pad_total - pad_total // 2
    pt, pb = pad_total // 2, pad_total - pad_total // 2
    x = layers.ZeroPadding2D(((pt, pb), (pl, pr)),
                             name=f"{name}_pad")(x)
    x = layers.DepthwiseConv2D(k, strides=1, padding="valid",
                               use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(epsilon=1e-5, name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def _mv2_block(x, in_ch, out_ch, kernel_size, expansion, use_residual, name):
    """1x1 expand -> kxk DW -> 1x1 project (+residual si use_residual)."""
    hidden = out_ch * expansion
    use_res = use_residual and in_ch == out_ch

    y = _conv_bn_relu(x, hidden, k=1, name=f"{name}_expand")
    y = _same_pad_dw(y, kernel_size, name=f"{name}_dw")
    y = layers.Conv2D(out_ch, 1, padding="valid", use_bias=False,
                      name=f"{name}_project_conv")(y)
    y = layers.BatchNormalization(epsilon=1e-5,
                                  name=f"{name}_project_bn")(y)
    if use_res:
        y = layers.Add(name=f"{name}_add")([y, x])
    return layers.ReLU(name=f"{name}_out_relu")(y)


def _branch(x, cfg, expansion, name):
    """MV2 -> Pool -> MV2 -> Pool -> MV2."""
    (k1, c1), (k2, c2), (k3, c3) = cfg
    cin = x.shape[-1]
    x = _mv2_block(x, cin, c1, k1, expansion, use_residual=False,
                   name=f"{name}_b1")
    x = layers.MaxPool2D(2, 2, name=f"{name}_p1")(x)
    x = _mv2_block(x, c1, c2, k2, expansion, use_residual=(c1 == c2),
                   name=f"{name}_b2")
    x = layers.MaxPool2D(2, 2, name=f"{name}_p2")(x)
    x = _mv2_block(x, c2, c3, k3, expansion, use_residual=(c2 == c3),
                   name=f"{name}_b3")
    return x


def build_microwd(input_shape=(768, 1024, 3), in_channels=3, expansion=3,
                  final_activation="softplus", drop_final_activation=False):
    """drop_final_activation=True devuelve los logits crudos antes del
    softplus/relu. Útil para cuantización int8: los logits tienen mayor
    rango → mejor resolución int8 → softplus se aplica fp32 en post-proceso."""
    if drop_final_activation:
        final_activation = "none"
    branches_cfg = [
        [(16, 12), (13, 12), (13, 6)],
        [(13, 24), (11, 24), (11, 6)],
        [(9, 16),  (7, 32),  (7, 8)],
        [(7, 20),  (5, 40),  (5, 10)],
    ]
    inp = keras.Input(shape=input_shape, name="image")
    b1 = _branch(inp, branches_cfg[0], expansion, "branch1")
    b2 = _branch(inp, branches_cfg[1], expansion, "branch2")
    b3 = _branch(inp, branches_cfg[2], expansion, "branch3")
    b4 = _branch(inp, branches_cfg[3], expansion, "branch4")
    x = layers.Concatenate(axis=-1, name="concat")([b1, b2, b3, b4])
    x = layers.Conv2D(1, 1, padding="valid", use_bias=True, name="out")(x)
    if final_activation == "softplus":
        x = layers.Activation("softplus", name="softplus")(x)
    elif final_activation == "relu":
        x = layers.ReLU(name="final_relu")(x)
    return keras.Model(inputs=inp, outputs=x, name="microwd_paper")


if __name__ == "__main__":
    m = build_microwd()
    m.summary(line_length=110)
    n = sum(v.numpy().size for v in m.trainable_variables) + \
        sum(v.numpy().size for v in m.non_trainable_variables)
    print(f"\ntotal params (incl. BN stats): {n:,}")

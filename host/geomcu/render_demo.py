"""
Render del demo visual: para cada imagen test, recorta una región 128x160
a su RESOLUCIÓN NATURAL (lo que vería 1 patch del tiling 8x8), corre el
modelo, muestra density map overlayed y el conteo predicho vs el ground
truth de puntos dentro del crop.

Output: docs/plots/geomcu_demo.gif + docs/plots/geomcu_demo_fN.png

El modelo .tflite usado es exactamente el mismo que está flasheado en el
ESP32-S3 (microwd_paper_logits_int8_patch_128x160.tflite). El count predicho
aquí es el que da el modelo en HOST; el chip da un count sistemáticamente
~27% más bajo por el rounding distinto de esp-nn en DWConv grandes,
documentado en firmware/geomcu_count/README.md.

Por qué crop y no resize: las imágenes ShanghaiTech son 768x1024. Resizear
a 128x160 mete a la gente en 4x5 pixeles y el modelo no detecta casi nada.
Cada patch del deploy 8x8 ve la imagen a su resolución natural, esa es la
condición que el modelo fue entrenado para resolver.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm

import tensorflow as tf


def crop_input(img_path: Path, ann_path: Path, crop_xy: tuple[int, int]
               ) -> tuple[Image.Image, np.ndarray, int]:
    """Crop 128x160 desde (x, y) en la imagen original.
    Devuelve (PIL crop, int8 array, gt_count dentro del crop)."""
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    x, y = crop_xy
    x = max(0, min(x, W - 160))
    y = max(0, min(y, H - 128))
    crop = pil.crop((x, y, x + 160, y + 128))
    arr = (np.asarray(crop, dtype=np.int16) - 128).astype(np.int8)
    pts = np.load(ann_path).astype(np.float32)
    in_crop = ((pts[:, 0] >= x) & (pts[:, 0] < x + 160) &
               (pts[:, 1] >= y) & (pts[:, 1] < y + 128))
    return crop, arr, int(in_crop.sum())


def softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def render_frame(rgb: Image.Image, density: np.ndarray, pred: float,
                 gt: int, img_id: str) -> Image.Image:
    """Compone frame: input 128x160 + heatmap overlay + banner texto.
    Resize todo a 480x384 para que se vea bien en GitHub."""
    target_w, target_h = 480, 384

    # density 32x40 -> upscale a 128x160 con interpolación nearest,
    # despues a target.
    d_norm = density / max(density.max(), 1e-6)
    d_color = (cm.jet(d_norm)[:, :, :3] * 255).astype(np.uint8)  # 32x40x3
    d_pil = Image.fromarray(d_color).resize((160, 128), Image.BILINEAR)

    base = rgb.resize((target_w, target_h - 60), Image.BILINEAR)
    heat = d_pil.resize((target_w, target_h - 60), Image.BILINEAR)
    blended = Image.blend(base, heat, alpha=0.45)

    canvas = Image.new("RGB", (target_w, target_h), (15, 15, 15))
    canvas.paste(blended, (0, 0))

    draw = ImageDraw.Draw(canvas, "RGBA")
    try:
        font_big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except IOError:
        font_big = font_sm = ImageFont.load_default()

    draw.rectangle([(0, target_h - 60), (target_w, target_h)],
                   fill=(0, 0, 0, 230))
    draw.text((12, target_h - 56), img_id, fill=(220, 220, 220), font=font_sm)
    label_pred = f"pred {pred:>6.1f}"
    label_gt = f"gt {gt:>3d}"
    draw.text((12, target_h - 32), label_pred,
              fill=(120, 220, 120), font=font_big)
    draw.text((220, target_h - 32), label_gt,
              fill=(200, 200, 200), font=font_big)
    return canvas


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="~/work/geomcu-deploy")
    # 8 (id, x, y) tuplas: crop manual donde haya gente.
    # ShanghaiTech Part B son 768H x 1024W. Centro denso típico: y~300, x~400.
    p.add_argument("--ids", nargs="+",
                   default=["IMG_1:400,300",
                            "IMG_10:400,300",
                            "IMG_50:300,250",
                            "IMG_75:400,250",
                            "IMG_100:400,300",
                            "IMG_111:400,250",
                            "IMG_150:400,300",
                            "IMG_200:400,300"])
    p.add_argument("--out-dir", default="/home/dominguez/OnCampus/MCUNetESP/docs/plots")
    p.add_argument("--frame-ms", type=int, default=1800)
    args = p.parse_args()

    ws = Path(os.path.expanduser(args.workspace))
    tflite_path = ws / "tflite" / "microwd_paper_logits_int8_patch_128x160.tflite"
    images_dir = ws / "dataset" / "part_B" / "test_data" / "images"
    ann_dir = ws / "dataset" / "part_B" / "test_data" / "annotations"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=8)
    inter.allocate_tensors()
    ind = inter.get_input_details()[0]
    outd = inter.get_output_details()[0]
    out_scale = float(outd["quantization_parameters"]["scales"][0])
    out_zp = int(outd["quantization_parameters"]["zero_points"][0])

    frames = []
    for spec in args.ids:
        sid, xy = spec.split(":")
        cx, cy = (int(v) for v in xy.split(","))
        rgb, x, gt = crop_input(images_dir / f"{sid}.jpg",
                                ann_dir / f"{sid}.npy", (cx, cy))
        inter.set_tensor(ind["index"], x[None])
        inter.invoke()
        out_int = inter.get_tensor(outd["index"])[0, ..., 0]
        logits = (out_int.astype(np.float32) - out_zp) * out_scale
        density = softplus(logits)
        pred = float(density.sum())
        label = f"{sid}  crop@({cx},{cy})"
        print(f"{sid:<8} crop({cx},{cy})  pred={pred:>6.2f}  gt={gt:>3d}  "
              f"density max={density.max():.3f}", flush=True)
        frame = render_frame(rgb, density, pred, gt, label)
        frames.append(frame)

    out_gif = out_dir / "geomcu_demo.gif"
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=args.frame_ms, loop=0)
    print(f"\nwrote {out_gif} ({out_gif.stat().st_size // 1024} KB)")
    for i, f in enumerate(frames):
        f.save(out_dir / f"geomcu_demo_f{i+1}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

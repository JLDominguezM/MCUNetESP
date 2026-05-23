#!/usr/bin/env python3
"""
Recibe N frames consecutivos del firmware camera_test (en modo multi-frame),
corre mcunet-vww2 sobre cada uno en host TFLite, y compone un GIF visual
con la imagen + overlay de la predicción (verde si PERSON, rojo si no-pers).

Uso:
    python host/decode_camera_frames_multi.py /dev/ttyACM0
"""

from __future__ import annotations
import base64
import io
import os
import re
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import serial
from PIL import Image, ImageDraw, ImageFont
from tensorflow.lite.python.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL = ROOT / "models" / "mcunet-vww2.tflite"


def decode_rgb565(data: bytes, w: int, h: int) -> Image.Image:
    arr = np.frombuffer(data, dtype=">u2").reshape(h, w)
    r = ((arr >> 11) & 0x1F) << 3
    g = ((arr >> 5) & 0x3F) << 2
    b = (arr & 0x1F) << 3
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def receive_frames(port: str, timeout_s: float = 45) -> list[Image.Image]:
    ser = serial.Serial()
    ser.port = port; ser.baudrate = 115200; ser.timeout = 0.5
    ser.dtr = False; ser.rts = False
    ser.open()
    ser.rts = True; time.sleep(0.1); ser.rts = False

    frames: list[Image.Image] = []
    capturing = False
    b64_lines: list[str] = []
    header: dict = {}
    buf = b""
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        chunk = ser.read(4096)
        if not chunk:
            continue
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode("utf-8", "replace").rstrip("\r")
            if "FRAME_B64:" not in text:
                try:
                    print(text, flush=True)
                except (BrokenPipeError, OSError):
                    pass  # downstream pipe closed; keep capturing silently
            if "FRAME_BEGIN" in text:
                m = re.search(r"fmt=(\d+) w=(\d+) h=(\d+) len=(\d+)", text)
                if m:
                    header = {"fmt": int(m.group(1)), "w": int(m.group(2)),
                              "h": int(m.group(3)), "len": int(m.group(4))}
                    capturing = True
                    b64_lines = []
            elif "FRAME_END" in text and capturing:
                capturing = False
                data = base64.b64decode("".join(b64_lines))
                w, h = header["w"], header["h"]
                if header["fmt"] == 0:
                    img = decode_rgb565(data, w, h)
                elif header["fmt"] == 4:
                    img = Image.open(io.BytesIO(data))
                else:
                    continue
                # No rotamos — el sensor del XIAO Sense entrega frames en
                # orientación natural cuando la placa está apoyada con la
                # cámara al frente del usuario.
                frames.append(img)
                print(f"  >>> frame {len(frames)} received ({w}x{h})")
            elif capturing and text.startswith("FRAME_B64:"):
                b64_lines.append(text.split("FRAME_B64:", 1)[1].strip())
    ser.close()
    return frames


def predict_vww(img: Image.Image, interp: Interpreter) -> tuple[str, int, int]:
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    _, h, w, _ = ind["shape"]
    resized = img.convert("RGB").resize((w, h), Image.BILINEAR)
    arr = (np.asarray(resized, dtype=np.int16) - 128).astype(np.int8)
    arr = arr[None, ...]
    interp.set_tensor(ind["index"], arr)
    interp.invoke()
    out = interp.get_tensor(outd["index"])[0]
    s_no, s_yes = int(out[0]), int(out[1])
    label = "PERSON" if s_yes > s_no else "no-pers"
    return label, s_no, s_yes


def annotate(img: Image.Image, label: str, s_no: int, s_yes: int,
             idx: int, total: int) -> Image.Image:
    out = img.resize((480, 480), Image.NEAREST)
    draw = ImageDraw.Draw(out, "RGBA")
    try:
        font_big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except IOError:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    color = (60, 200, 80, 230) if label == "PERSON" else (210, 60, 60, 230)
    draw.rectangle([(0, 0), (480, 70)], fill=color)
    draw.text((12, 14), label, fill=(255, 255, 255), font=font_big)
    draw.text((350, 22), f"{idx+1}/{total}",
              fill=(255, 255, 255), font=font_small)
    draw.rectangle([(0, 430), (480, 480)], fill=(0, 0, 0, 180))
    draw.text((12, 440),
              f"scores=[no-pers:{s_no:+d}  person:{s_yes:+d}]",
              fill=(255, 255, 255), font=font_small)
    return out


def main(port: str) -> int:
    print(f"# receiving frames from {port} ...")
    # Cada frame en RGB565 240x240 = 115200 bytes → ~16 s a 115200 baud (base64
    # de 6/8 ratio). 5 frames × (16 + 1.5) ≈ 90 s. Damos margen.
    frames = receive_frames(port, timeout_s=120)
    if not frames:
        print("ERROR: no frames received")
        return 1
    print(f"\n# received {len(frames)} frames; running inference ...")
    interp = Interpreter(model_path=str(MODEL))
    interp.allocate_tensors()
    annotated: list[Image.Image] = []
    for i, img in enumerate(frames):
        label, s_no, s_yes = predict_vww(img, interp)
        print(f"  frame {i+1}: {label}  scores=[{s_no:+d}, {s_yes:+d}]")
        annotated.append(annotate(img, label, s_no, s_yes, i, len(frames)))
    out = OUT_DIR / "demo_visual.gif"
    annotated[0].save(out, save_all=True, append_images=annotated[1:],
                      duration=1500, loop=0)
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")
    for i, im in enumerate(annotated):
        im.save(OUT_DIR / f"demo_visual_f{i+1}.png")
    return 0


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    sys.exit(main(port))

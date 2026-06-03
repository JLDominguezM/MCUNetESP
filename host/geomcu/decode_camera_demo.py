"""
Recibe N frames RGB565 base64 capturados por geomcu_count_live, junto con
las líneas RESULT del propio chip (count, latency, max_density), y arma
un GIF mostrando el frame de cámara con un recuadro al crop 160x128 + el
count y la latencia reportados por el ESP32-S3.

Uso:
    python decode_camera_demo.py /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import serial
from PIL import Image, ImageDraw, ImageFont


def decode_rgb565(data: bytes, w: int, h: int) -> Image.Image:
    arr = np.frombuffer(data, dtype=">u2").reshape(h, w)
    r = ((arr >> 11) & 0x1F) << 3
    g = ((arr >> 5) & 0x3F) << 2
    b = (arr & 0x1F) << 3
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def receive(port: str, timeout_s: float = 240) -> list[tuple[Image.Image, dict]]:
    ser = serial.Serial()
    ser.port = port; ser.baudrate = 115200; ser.timeout = 0.5
    ser.dtr = False; ser.rts = False
    ser.open()
    ser.rts = True; time.sleep(0.1); ser.rts = False

    frames: list[tuple[Image.Image, dict]] = []
    pending_img: Image.Image | None = None
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
                    pass
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
                if header["fmt"] == 0:
                    pending_img = decode_rgb565(data, header["w"], header["h"])
                else:
                    pending_img = Image.open(io.BytesIO(data))
                print(f"  >>> frame {len(frames)+1} received "
                      f"({header['w']}x{header['h']})")
            elif capturing and text.startswith("FRAME_B64:"):
                b64_lines.append(text.split("FRAME_B64:", 1)[1].strip())
            elif "RESULT:" in text and pending_img is not None:
                m = re.search(r"latency_ms=(\d+) sum_full=([\d\.]+) "
                              r"sum_interior=([\d\.]+) max=([\d\.]+)", text)
                if m:
                    res = {"latency_ms": int(m.group(1)),
                           "sum_full": float(m.group(2)),
                           "sum_interior": float(m.group(3)),
                           "max": float(m.group(4))}
                    frames.append((pending_img, res))
                    pending_img = None
                    if len(frames) >= 4:
                        ser.close()
                        return frames
    ser.close()
    return frames


CROP_W, CROP_H = 160, 128


def annotate(img: Image.Image, res: dict, idx: int, total: int
             ) -> Image.Image:
    """Renderiza frame 240x240 escalado a 480x480 con recuadro al crop
    centrado 160x128 y banner de count + latency."""
    target = 480
    base = img.resize((target, target), Image.NEAREST)
    canvas = Image.new("RGB", (target, target + 80), (15, 15, 15))
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    try:
        font_big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except IOError:
        font_big = font_sm = ImageFont.load_default()

    cam_w, cam_h = img.size
    scale = target / cam_w
    x0 = int(((cam_w - CROP_W) // 2) * scale)
    y0 = int(((cam_h - CROP_H) // 2) * scale)
    x1 = x0 + int(CROP_W * scale)
    y1 = y0 + int(CROP_H * scale)
    draw.rectangle([(x0, y0), (x1, y1)], outline=(40, 220, 100, 255), width=4)
    draw.text((x0 + 6, y0 + 4), "model input 160x128",
              fill=(40, 220, 100), font=font_sm)

    draw.rectangle([(0, target), (target, target + 80)],
                   fill=(0, 0, 0, 230))
    label = (f"count {res['sum_full']:.2f}    "
             f"{res['latency_ms']/1000:.1f}s    {idx+1}/{total}")
    draw.text((14, target + 14), label,
              fill=(255, 255, 255), font=font_big)
    draw.text((14, target + 50),
              f"max density {res['max']:.3f}    interior {res['sum_interior']:.2f}",
              fill=(180, 180, 180), font=font_sm)
    return canvas


def main(port: str) -> int:
    out_dir = Path("/home/dominguez/OnCampus/MCUNetESP/docs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# escuchando {port}, espero hasta 240s o 4 frames con RESULT")
    items = receive(port, timeout_s=240)
    if not items:
        print("ERROR: no frames con RESULT recibidos")
        return 1
    print(f"\n# recibí {len(items)} (frame, result). Renderizando GIF...")
    frames = [annotate(img, res, i, len(items))
              for i, (img, res) in enumerate(items)]
    out_gif = out_dir / "geomcu_chip_demo.gif"
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=2500, loop=0)
    for i, f in enumerate(frames):
        f.save(out_dir / f"geomcu_chip_demo_f{i+1}.png")
    print(f"wrote {out_gif} ({out_gif.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    sys.exit(main(port))

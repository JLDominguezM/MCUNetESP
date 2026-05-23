#!/usr/bin/env python3
"""
Lee la salida serial de firmware/camera_test/, busca el bloque FRAME_BEGIN..FRAME_END
y decodifica la imagen capturada por la OV2640.

Uso:
    python host/decode_camera_frame.py /dev/ttyACM0 [seconds=30]
"""

from __future__ import annotations
import base64
import re
import sys
import time
from pathlib import Path

import serial
import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(port: str, seconds: float) -> int:
    ser = serial.Serial()
    ser.port = port; ser.baudrate = 115200; ser.timeout = 0.5
    ser.dtr = False; ser.rts = False
    ser.open()
    ser.rts = True; time.sleep(0.1); ser.rts = False

    t_end = time.time() + seconds
    buf = b""
    capturing = False
    b64_lines: list[str] = []
    header: dict[str, int] = {}
    while time.time() < t_end:
        chunk = ser.read(4096)
        if not chunk:
            continue
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode("utf-8", "replace").rstrip("\r")
            if "FRAME_B64:" not in text:
                print(text)
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
                print(f"\n>>> decoded {len(data)} bytes (expected ~{header.get('len', '?')})")
                fmt = header.get("fmt", -1)
                w, h = header.get("w", 0), header.get("h", 0)
                # esp32-camera pixformat_t enum: 0=RGB565, 1=YUV422, 2=YUV420,
                # 3=GRAYSCALE, 4=JPEG, 5=RGB888, 6=RAW, 7=RGB444, 8=RGB555.
                if fmt == 4:  # JPEG
                    out = OUT_DIR / "sample_camera_frame.jpg"
                    out.write_bytes(data)
                    print(f"saved {out} (JPEG)")
                    img = Image.open(out)
                    png = OUT_DIR / "sample_camera_frame.png"
                    img.save(png)
                    print(f"converted to {png}")
                elif fmt == 0:  # RGB565 (big-endian en el bus DVP)
                    arr = np.frombuffer(data, dtype=">u2").reshape(h, w)
                    r = ((arr >> 11) & 0x1F) << 3
                    g = ((arr >> 5) & 0x3F) << 2
                    b = (arr & 0x1F) << 3
                    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
                    out = OUT_DIR / "sample_camera_frame.png"
                    Image.fromarray(rgb).save(out)
                    print(f"saved {out} (RGB565 BE → RGB888)")
                elif fmt == 3:  # GRAYSCALE
                    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
                    out = OUT_DIR / "sample_camera_frame.png"
                    Image.fromarray(arr, mode="L").save(out)
                    print(f"saved {out} (GRAYSCALE)")
                else:
                    out = OUT_DIR / f"sample_camera_frame.fmt{fmt}.bin"
                    out.write_bytes(data)
                    print(f"unknown fmt={fmt}, saved raw to {out}")
                ser.close()
                return 0
            elif capturing and text.startswith("FRAME_B64:"):
                b64_lines.append(text.split("FRAME_B64:", 1)[1].strip())
    ser.close()
    print("timeout — no FRAME captured")
    return 1


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    sys.exit(main(port, secs))

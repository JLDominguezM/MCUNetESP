#!/usr/bin/env python3
"""
Convierte una imagen a un C++ array RGB int8 con zero_point = -128
(formato esperado por modelos TFLite int8 cuantizados de MCUNet/MobileNetV2).

Uso:
    python image_to_c.py <input.jpg> <W>x<H> <out.cc> <symbol_name>

Ejemplo:
    python image_to_c.py sample_images/person_det.jpg 144x144 ../firmware/vww_demo/main/test_image.cc g_test_image_vww
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print("usage: image_to_c.py <input> <WxH> <out.cc> <symbol>", file=sys.stderr)
        return 2
    src, size_str, out_path, sym = argv[1], argv[2], argv[3], argv[4]
    w, h = (int(x) for x in size_str.lower().split("x"))

    img = Image.open(src).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.int16) - 128  # quantize to int8 (-128..127)
    arr = arr.astype(np.int8).flatten()

    body = ", ".join(str(int(x)) for x in arr.tolist())
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"// Auto-generated from {src} ({w}x{h} RGB int8, zp=-128).\n")
        f.write("#include <cstdint>\n\n")
        f.write('extern "C" {\n\n')
        f.write(f"extern __attribute__((used))\n")
        f.write(f"alignas(16) const int8_t {sym}[{w*h*3}] = {{ {body} }};\n\n")
        f.write(f"extern __attribute__((used))\n")
        f.write(f"const int {sym}_w = {w};\n")
        f.write(f"extern __attribute__((used))\n")
        f.write(f"const int {sym}_h = {h};\n\n")
        f.write('}  // extern "C"\n')
    print(f"wrote {out} ({w}x{h}, {w*h*3} bytes payload, sym={sym})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

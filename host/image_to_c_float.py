#!/usr/bin/env python3
"""
Convierte una imagen a un C++ array RGB float32 en rango [-1, +1]
(formato esperado por modelos TFLite float32 como mcunet-person-det).

Uso:
    python image_to_c_float.py <input.jpg> <WxH> <out.cc> <symbol>
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print("usage: image_to_c_float.py <input> <WxH> <out.cc> <symbol>", file=sys.stderr)
        return 2
    src, size_str, out_path, sym = argv[1], argv[2], argv[3], argv[4]
    w, h = (int(x) for x in size_str.lower().split("x"))

    img = Image.open(src).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0
    flat = arr.flatten()

    body = ", ".join(f"{float(x):.5f}f" for x in flat.tolist())
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"// Auto-generated from {src} ({w}x{h} RGB float32, range [-1, +1]).\n")
        f.write("#include <cstdint>\n\n")
        f.write('extern "C" {\n\n')
        f.write(f"extern __attribute__((used))\n")
        f.write(f"alignas(16) const float {sym}[{w*h*3}] = {{ {body} }};\n\n")
        f.write(f"extern __attribute__((used))\n")
        f.write(f"const int {sym}_w = {w};\n")
        f.write(f"extern __attribute__((used))\n")
        f.write(f"const int {sym}_h = {h};\n\n")
        f.write('}  // extern "C"\n')
    print(f"wrote {out} ({w}x{h} float32, {w*h*3*4} bytes payload, sym={sym})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

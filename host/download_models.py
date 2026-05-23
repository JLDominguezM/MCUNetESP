#!/usr/bin/env python3
"""
Descarga los modelos preentrenados MCUNet (.tflite int8) al directorio ../models/.

Nombres y URLs verificados contra mcunet/model_zoo.py (commit 9c164f8 del repo
mit-han-lab/mcunet). url_base = https://hanlab18.mit.edu/projects/tinyml/mcunet/release/

Modelos:
  - mcunet-vww2          -> mcunet-320kb-1mb_vww.tflite           (HITO 1)
  - mcunet-in0           -> mcunet-10fps_imagenet.tflite          (HITO 2)
  - mcunet-in2           -> mcunet-256kb-1mb_imagenet.tflite      (alternativa más chica)
  - person-det           -> person-det.tflite                     (HITO 3)
  - mbv2-w0.35           -> mbv2-w0.35-r144_imagenet.tflite       (HITO 4 baseline)
  - proxyless-w0.3       -> proxyless-w0.3-r176_imagenet.tflite   (HITO 4 baseline alt)
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

URL_BASE = "https://hanlab18.mit.edu/projects/tinyml/mcunet/release/"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# alias -> (filename_remote_basename, filename_local)
MODELS: dict[str, tuple[str, str]] = {
    "mcunet-vww2":     ("mcunet-320kb-1mb_vww",          "mcunet-vww2.tflite"),
    "mcunet-in0":      ("mcunet-10fps_imagenet",         "mcunet-in0.tflite"),
    "mcunet-in2":      ("mcunet-256kb-1mb_imagenet",     "mcunet-in2.tflite"),
    "person-det":      ("person-det",                    "mcunet-person-det.tflite"),
    "mbv2-w0.35":      ("mbv2-w0.35-r144_imagenet",      "mbv2-w0.35.tflite"),
    "proxyless-w0.3":  ("proxyless-w0.3-r176_imagenet",  "proxyless-w0.3.tflite"),
}


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {MODELS_DIR}")

    failed: list[str] = []
    for alias, (remote, local) in MODELS.items():
        dst = MODELS_DIR / local
        url = URL_BASE + remote + ".tflite"
        if dst.exists() and dst.stat().st_size > 1000:
            print(f"  [skip] {local} (already present, {dst.stat().st_size} bytes)")
            continue
        print(f"  [get ] {alias}  <-  {url}")
        try:
            urlretrieve(url, dst)
            print(f"         OK ({dst.stat().st_size} bytes)")
        except Exception as exc:
            print(f"         FAILED: {exc}")
            failed.append(alias)
            if dst.exists():
                dst.unlink()

    print()
    if failed:
        print(f"Algunos modelos fallaron: {', '.join(failed)}")
        print(f"URL base usada: {URL_BASE}")
        print("Si todas fallan, prueba abrir la URL en un navegador para ver si el host responde.")
        return 1
    print("Todos los modelos descargados correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

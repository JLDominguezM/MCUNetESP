"""
Sanity check del forward float32 contra el run de referencia
(microwd_paper_original_..._seed7, best_mae.pt).

Validación:
  - param count == 85,057
  - state_dict carga con strict=True (topología + nombres exactos)
  - pred_count local vs detailed_eval/per_image_predictions.csv del repo

Esperar offset sistemático ~0.1% relativo (las predicciones de referencia
se generaron en GPU con TF32; nosotros corremos en CPU FP32 puro). Lo que
importa es que sea consistente y chico, los pesos son correctos.

Uso:
    python forward_check.py
    python forward_check.py --ids IMG_1 IMG_10 IMG_100
    python forward_check.py --workspace ~/work/geomcu-deploy --rel-tol 0.005
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model_microwd import MiCrowdNetPaperFullFrame, count_params


def load_image(path: Path) -> torch.Tensor:
    """Replica el preprocesado de ShanghaiFullResDataset (resize_mode='none',
    out_stride=4). El raw es 768x1024 → ya múltiplo de 4 → no se aplica pad.
    Equivalente a cv2.imread + cvtColor(BGR2RGB) + /255 cuando no hay resize."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    return t


def load_expected_predictions(csv_path: Path) -> dict[str, float]:
    out = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["id"]] = float(row["pred_count"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="~/work/geomcu-deploy",
                        help="dir con run/ y dataset/")
    parser.add_argument("--ids", nargs="+", default=["IMG_1", "IMG_10", "IMG_100",
                                                    "IMG_101", "IMG_102"])
    parser.add_argument("--rel-tol", type=float, default=0.005,
                        help="tolerancia relativa sobre pred_count (0.5%)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ws = Path(os.path.expanduser(args.workspace))
    ckpt_path = ws / "run" / "best_mae.pt"
    images_dir = ws / "dataset" / "part_B" / "test_data" / "images"
    csv_path = ws / "run" / "detailed_eval" / "per_image_predictions.csv"

    print(f"workspace : {ws}")
    print(f"checkpoint: {ckpt_path}")
    print(f"images    : {images_dir}")
    print(f"reference : {csv_path}\n")

    model = MiCrowdNetPaperFullFrame()
    print(f"params : {count_params(model):,}  (expected 85,057)")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()

    expected = load_expected_predictions(csv_path)

    print(f"{'id':<10} {'pred':>10} {'expected':>10} {'rel_err':>10}  ok?")
    print("-" * 52)
    n_ok = 0
    diffs = []
    for sid in args.ids:
        img_t = load_image(images_dir / f"{sid}.jpg").to(args.device)
        with torch.no_grad():
            d = model(img_t)
        pred = float(d.sum().item())
        ref = expected.get(sid, float("nan"))
        rel = (pred - ref) / ref if ref else float("nan")
        ok = abs(rel) < args.rel_tol
        n_ok += int(ok)
        diffs.append(rel)
        mark = "OK" if ok else "MISMATCH"
        print(f"{sid:<10} {pred:>10.4f} {ref:>10.4f} {rel*100:>+9.3f}%  {mark}")

    if diffs:
        import statistics
        print(f"\nrel_err mean={statistics.mean(diffs)*100:+.3f}%  "
              f"stdev={statistics.pstdev(diffs)*100:.3f}%")
    print(f"{n_ok}/{len(args.ids)} match dentro de rel_tol={args.rel_tol*100:.2f}%")
    return 0 if n_ok == len(args.ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())

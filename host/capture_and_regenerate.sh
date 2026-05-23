#!/usr/bin/env bash
#
# Script "all-in-one" para regenerar la documentación con una imagen capturada
# por la cámara OV2640 del XIAO, en lugar de la imagen `person_det.jpg`.
#
# Pre-condición: docs/plots/sample_camera_frame.png debe existir.
# Eso lo produce `host/decode_camera_frame.py` después de flashear `camera_test/`.
#
# Flujo:
#   1. Verifica que sample_camera_frame.png existe
#   2. Copia la imagen a host/sample_images/captured_frame.png (como nueva imagen de test)
#   3. Re-bakea test_image.cc en los 5 firmware con la imagen capturada
#   4. Re-flashea y captura nuevas predicciones (logs reales) para cada uno
#   5. Re-genera todos los plots con la imagen real
#
# Uso:
#   ./host/capture_and_regenerate.sh
#
# Toma ~10-15 minutos por los re-flashes.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CAPTURED="docs/plots/sample_camera_frame.png"
TEST_IMAGE="host/sample_images/captured_frame.png"

# ─── 1. Verify capture exists ──────────────────────────────────────────────────
if [[ ! -f "$CAPTURED" ]]; then
    echo "ERROR: $CAPTURED no existe."
    echo
    echo "Primero captura un frame con:"
    echo "  cd firmware/camera_test && idf.py -p /dev/ttyACM0 flash && cd ../.."
    echo "  python host/decode_camera_frame.py /dev/ttyACM0 30"
    echo
    echo "Si la cámara sigue fallando, ver docs/CAMERA_FIX.md."
    exit 1
fi

echo "==> Encontrado: $CAPTURED ($(stat -c %s "$CAPTURED") bytes)"
cp "$CAPTURED" "$TEST_IMAGE"
echo "==> Copiado a $TEST_IMAGE"

# ─── 2. Activate conda env ─────────────────────────────────────────────────────
# Permite override del path de conda con CONDA_BASE; default a ~/miniconda3.
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mcunet

# ─── 3. Re-bake test_image.cc in every firmware ────────────────────────────────
echo
echo "==> Re-bakeando test_image en cada firmware ..."

# vww_demo + benchmark_mbv2:    144x144 int8 → g_test_image_vww / g_test_image_in
python host/image_to_c.py "$TEST_IMAGE" 144x144 firmware/vww_demo/main/test_image.cc        g_test_image_vww
python host/image_to_c.py "$TEST_IMAGE" 144x144 firmware/benchmark_mbv2/main/test_image.cc  g_test_image_in

# imagenet_demo:                 160x160 int8 → g_test_image_in
python host/image_to_c.py "$TEST_IMAGE" 160x160 firmware/imagenet_demo/main/test_image.cc   g_test_image_in

# benchmark_proxyless:           176x176 int8 → g_test_image_in
python host/image_to_c.py "$TEST_IMAGE" 176x176 firmware/benchmark_proxyless/main/test_image.cc g_test_image_in

# person_detect_demo:            160x128 float32 → g_test_image_pd
python host/image_to_c_float.py "$TEST_IMAGE" 160x128 firmware/person_detect_demo/main/test_image.cc g_test_image_pd

echo "==> Done bakeando."

# ─── 4. Source ESP-IDF ─────────────────────────────────────────────────────────
source ~/esp/esp-idf/export.sh > /dev/null 2>&1

# ─── 5. Build + flash + capture predictions for each firmware ──────────────────
declare -a FIRMWARES=("vww_demo" "imagenet_demo" "person_detect_demo" "benchmark_mbv2" "benchmark_proxyless")
declare -a TAGS=("vww" "imagenet" "person_det" "mbv2" "proxyless")

for i in "${!FIRMWARES[@]}"; do
    fw="${FIRMWARES[$i]}"
    tag="${TAGS[$i]}"
    echo
    echo "==> [$fw] build + flash + capture ..."
    pushd "firmware/$fw" > /dev/null
    idf.py build > "/tmp/${fw}_rebake.log" 2>&1
    idf.py -p /dev/ttyACM0 flash >> "/tmp/${fw}_rebake.log" 2>&1
    popd > /dev/null
    sleep 1
    python host/monitor.py /dev/ttyACM0 12 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | grep -E "${tag}:|top5|PROFILER|PERSON|no-pers|model OK|free heap" \
        > "docs/logs/${fw}.log"
    echo "    wrote docs/logs/${fw}.log"
done

# ─── 6. Reproduce on host and regen plots ──────────────────────────────────────
echo
echo "==> Reproduciendo en host y regenerando plots ..."
python host/reproduce_host_inference.py > docs/logs/host_reproduction.log 2>&1
python host/generate_plots.py
python host/visualize_inference.py

echo
echo "==> DONE. Imagen capturada ($TEST_IMAGE) ahora es la base de toda la doc."
echo "==> Revisa: docs/REPORT.md, docs/findings.md, docs/plots/, docs/logs/"

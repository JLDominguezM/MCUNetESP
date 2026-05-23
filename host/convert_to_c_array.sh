#!/usr/bin/env bash
# Convierte un .tflite a un C array para embeber en firmware ESP-IDF.
#
# Uso:
#   ./convert_to_c_array.sh ../models/mcunet-vww2.tflite ../firmware/vww_demo/main/model_data.cc
#
# El nombre del símbolo se deriva del nombre base del archivo (.tflite quitado).

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <input.tflite> <output.cc>" >&2
    exit 2
fi

IN="$1"
OUT="$2"
SYM=$(basename "$IN" .tflite | tr '-' '_' | tr '.' '_')

if ! command -v xxd >/dev/null 2>&1; then
    echo "error: xxd not found. install vim-common or xxd." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"

{
    echo "// Auto-generated from $IN — do not edit by hand."
    echo "#include <cstdint>"
    echo
    echo "// NOTA: \`const\` en C++ tiene internal linkage por defecto, así que"
    echo "// hay que declarar \`extern\` explícitamente para que main.cc pueda enlazar."
    echo "// El \`extern \"C\"\` solo afecta name mangling, no el linkage."
    echo "extern \"C\" {"
    echo
    echo "extern __attribute__((used))"
    echo "alignas(16) const unsigned char g_${SYM}[] = {"
    xxd -i < "$IN" | sed 's/^/    /'
    echo "};"
    echo
    SIZE=$(stat -c%s "$IN")
    echo "extern __attribute__((used))"
    echo "const unsigned int g_${SYM}_len = ${SIZE};"
    echo
    echo "}  // extern \"C\""
} > "$OUT"

echo "wrote $OUT (symbol g_${SYM}, ${SIZE} bytes)"

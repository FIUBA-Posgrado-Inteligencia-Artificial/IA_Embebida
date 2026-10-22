#!/usr/bin/env bash
# install_model.sh <source> — copia model.c + golden_data.h + model_meta.json
# desde tools/rt_export/onnx2c/out/<source>/ a main/ (gitignoreados).
#
# Si el input es >= 64px (buffers intermedios > DRAM interna), anota los
# buffers estáticos de model.c (tu0/tu1/...) con EXT_RAM_BSS_ATTR para que
# vayan a PSRAM (requiere CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY=y,
# ya en sdkconfig.defaults).
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE="${1:?uso: install_model.sh <source>  (p.ej. depthwise_rgb32_pruned_activation_30_ft2)}"
OUT="../../tools/rt_export/onnx2c/out/$SOURCE"
[ -f "$OUT/model.c" ] || { echo "No existe $OUT/model.c — correr export_onnx2c.py --source $SOURCE"; exit 1; }

cp "$OUT/model.c" main/model.c
cp "$OUT/golden_data.h" main/golden_data.h
cp "$OUT/model_meta.json" main/model_meta.json

W=$(sed -n 's/#define GOLDEN_W \([0-9]*\)/\1/p' main/golden_data.h)
if [ "$W" -ge 64 ]; then
    echo "[install_model.sh] input ${W}px: moviendo buffers de model.c a PSRAM"
    sed -i 's/^static union tensor_union_\([0-9]*\) tu\([0-9]*\);/EXT_RAM_BSS_ATTR static union tensor_union_\1 tu\2;/' main/model.c
    sed -i '1i #include "esp_attr.h"' main/model.c
fi
echo "[install_model.sh] instalado: $SOURCE"

#!/usr/bin/env bash
# install_model.sh <source> — copia model_data.h + golden_data.h + model_meta.json
# desde tools/rt_export/tflite/out/<source>/ a main/ (gitignoreados).
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE="${1:?uso: install_model.sh <source>  (p.ej. depthwise_rgb32_pruned_activation_30_ft2)}"
OUT="../../tools/rt_export/tflite/out/$SOURCE"
[ -f "$OUT/model_data.h" ] || { echo "No existe $OUT/model_data.h — correr export_tflite.py --source $SOURCE"; exit 1; }

cp "$OUT/model_data.h" main/model_data.h
cp "$OUT/golden_data.h" main/golden_data.h
cp "$OUT/model_meta.json" main/model_meta.json
echo "[install_model.sh] instalado: $SOURCE"

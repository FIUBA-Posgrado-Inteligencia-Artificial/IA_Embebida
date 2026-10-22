#!/bin/bash
# Instala los headers generados por export_executorch.py en main/.
# Uso: ./install_model.sh <source>   (ej: depthwise_rgb32_pruned_activation_30_ft2)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$HERE/../../tools/rt_export/executorch/out/${1:?uso: $0 <source>}"
[ -f "$SRC_DIR/model_pte.h" ] || { echo "falta $SRC_DIR/model_pte.h (correr export_executorch.py --source $1)"; exit 1; }
cp "$SRC_DIR/model_pte.h" "$SRC_DIR/golden_data.h" "$HERE/main/"
echo "instalado: $1"

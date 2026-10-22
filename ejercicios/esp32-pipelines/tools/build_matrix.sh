#!/usr/bin/env bash
# Smoke: el firmware compila para todas las placas y combos clave de features.
# Uso: . ~/esp/esp-idf/export.sh && tools/build_matrix.sh
#
# -DSDKCONFIG=build_$b/sdkconfig aisla el sdkconfig por build dir: sin esto,
# idf.py usa por default el sdkconfig compartido en la raíz del proyecto
# (firmware/sdkconfig), así que buildear una placa después de otra con chip
# distinto (ej. esp32cam tras esp32s3) deja el build_$b/CMakeCache.txt de la
# primera desincronizado del sdkconfig top-level y el build siguiente falla
# con "Project sdkconfig ... was generated for target 'X', but CMakeCache.txt
# contains 'Y'". Descubierto corriendo la matriz completa seguida de rebuilds
# puntuales de esp32cam (combos de features).
set -e
cd "$(dirname "$0")/../firmware"
for b in esp32 esp32cam esp32s3 esp32s3cam; do
    echo "=== BOARD=$b ==="
    chip=esp32; case "$b" in esp32s3*) chip=esp32s3;; esac
    sdk="-DSDKCONFIG=build_$b/sdkconfig"
    [ -f "build_$b/sdkconfig" ] || idf.py -B "build_$b" "-DBOARD=$b" "$sdk" set-target "$chip"
    idf.py -B "build_$b" "-DBOARD=$b" "$sdk" build
done
echo "MATRIZ OK"

#!/usr/bin/env bash
# build.sh — clona y compila onnx2c (https://github.com/kraiskil/onnx2c).
#
# Deja el binario en tools/rt_export/onnx2c/onnx2c-src/build/onnx2c.
# Si el sistema no tiene protobuf (protoc + libprotobuf-dev), compila un
# protobuf estático local en protobuf-install/ (una sola vez, ~5 min).
#
# Todo lo generado (onnx2c-src/, protobuf-src/, protobuf-install/) está
# gitignoreado — correr este script es la forma de regenerarlo.
set -euo pipefail
cd "$(dirname "$0")"

ONNX2C_REPO=https://github.com/kraiskil/onnx2c
PROTOBUF_TAG=v21.12
JOBS=$(nproc)

# --- 1. clonar onnx2c (con submodules: onnx) --------------------------------
if [ ! -d onnx2c-src ]; then
    git clone --recursive "$ONNX2C_REPO" onnx2c-src
fi

# --- 2. protobuf: usar el del sistema si existe, sino build local -----------
CMAKE_EXTRA=()
if ! command -v protoc >/dev/null 2>&1; then
    if [ ! -x protobuf-install/bin/protoc ]; then
        echo "[build.sh] protoc no encontrado: compilando protobuf $PROTOBUF_TAG local..."
        if [ ! -d protobuf-src ]; then
            git clone --depth 1 --branch "$PROTOBUF_TAG" \
                https://github.com/protocolbuffers/protobuf protobuf-src
        fi
        cmake -S protobuf-src -B protobuf-src/build \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$PWD/protobuf-install" \
            -Dprotobuf_BUILD_TESTS=OFF \
            -Dprotobuf_BUILD_SHARED_LIBS=OFF \
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        cmake --build protobuf-src/build -j "$JOBS"
        cmake --install protobuf-src/build
    fi
    export PATH="$PWD/protobuf-install/bin:$PATH"
    CMAKE_EXTRA+=("-DCMAKE_PREFIX_PATH=$PWD/protobuf-install")
fi

# --- 3. build onnx2c ---------------------------------------------------------
cmake -S onnx2c-src -B onnx2c-src/build -DCMAKE_BUILD_TYPE=Release "${CMAKE_EXTRA[@]}"
cmake --build onnx2c-src/build -j "$JOBS" --target onnx2c

echo "[build.sh] OK: $PWD/onnx2c-src/build/onnx2c"
./onnx2c-src/build/onnx2c --help >/dev/null 2>&1 || true

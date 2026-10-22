# firmware_rt/executorch — runtime ExecuTorch portable (float) en ESP32-S3

Self-test por UART (formato `script06-bench.py`, runtime `executorch`) del
runtime portable de ExecuTorch corriendo un `.pte` float embebido, sin
delegates. Board: devkit ESP32-S3 N16R8 (16MB flash, 8MB PSRAM octal).

## Cómo se armó (spike 2026-08-14)

ExecuTorch v1.4.0 trae soporte upstream para ESP-IDF: preset CMake
`esp-baremetal` + `examples/espressif/`. Este proyecto usa ese camino
(enfoque "a" del plan): cross-compilar el runtime con el toolchain Xtensa
de ESP-IDF y linkear las `.a` desde un componente IDF propio.

### 1. Clonar executorch (gitignoreado, regenerable)

```bash
cd firmware_rt/executorch/third_party
git clone --depth 1 --branch v1.4.0 https://github.com/pytorch/executorch executorch
cd executorch
git submodule update --init --depth 1 \
    third-party/flatbuffers third-party/flatcc third-party/json \
    third-party/gflags \
    backends/xnnpack/third-party/FXdiv \
    backends/xnnpack/third-party/cpuinfo \
    backends/xnnpack/third-party/pthreadpool
```

(cpuinfo/pthreadpool/FXdiv/gflags los exige el CMake top-level aunque no se
use XNNPACK.)

### 2. Cross-compilar el runtime a Xtensa

```bash
. ~/esp/esp-idf/export.sh
cd firmware_rt/executorch/third_party/executorch
cmake --preset esp-baremetal -B cmake-out-esp \
    -DCMAKE_TOOLCHAIN_FILE=$IDF_PATH/tools/cmake/toolchain-esp32s3.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
    -DEXECUTORCH_OPTIMIZE_SIZE=ON \
    -DPYTHON_EXECUTABLE=$PWD/../../../tools/rt_export/executorch/.venv/bin/python
cmake --build cmake-out-esp -j$(nproc)
cmake --build cmake-out-esp --target install
```

**`EXECUTORCH_OPTIMIZE_SIZE=ON` es obligatorio**: con `-O2`, el gcc 13.2 de
Xtensa tira un *internal compiler error* (pass postreload) compilando
`kernels/portable/cpu/op_sum.cpp`. Con `-Os` compila todo.

El `PYTHON_EXECUTABLE` es el venv de `tools/rt_export/executorch` (lo usa el
codegen de registración de kernels; necesita torch instalado).

### 3. Exportar el modelo y buildear

```bash
uv run --project tools/rt_export/executorch python \
    tools/rt_export/executorch/export_executorch.py --source <source>
cd firmware_rt/executorch
./install_model.sh <source>       # copia model_pte.h + golden_data.h a main/
idf.py set-target esp32s3         # solo la primera vez
idf.py build                      # flash: idf.py -p /dev/ttyUSB0 flash monitor
```

## Detalles

- Kernels portables completos, linkeados con `--whole-archive`
  (`portable_ops_lib` registra por constructores estáticos). Sin selective
  build: binario ~4.4MB, entra de sobra en la partición app de 8MB.
- Input float32 NCHW normalizado; referencia = logits del runtime ET en PC
  (pybindings), tolerancia float `tol=0.01` en `selftest_runner`.
- Method allocator pool 2MB + planned buffers en PSRAM
  (`heap_caps_malloc(MALLOC_CAP_SPIRAM)` con fallback a heap interno).
- PAL: el default `posix` del runtime compila y corre sobre newlib de
  ESP-IDF (no hizo falta el `esp_pal.cpp` del ejemplo upstream).

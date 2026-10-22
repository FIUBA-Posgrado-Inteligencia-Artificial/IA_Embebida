# firmware_rt/tflm — runtime TFLite Micro (int8) en ESP32-S3

Proyecto ESP-IDF mínimo (sin cámara) que corre el self-test de golden images
sobre un modelo `.tflite` full-int8 con `espressif/esp-tflite-micro`
(MicroInterpreter). Usa el componente compartido `firmware_rt/common/`
(selftest_runner): imprime `ready`, y por UART `t` corre el self-test /
`c` una inferencia.

- Entrada: int8 **NHWC** ya cuantizada al input del modelo (las golden vienen
  así en `golden_data.h`, generadas por `tools/rt_export/tflite/export_tflite.py`).
- Referencia: logits int8 del interpreter TFLite de PC sobre el MISMO
  `.tflite` → modo int8 del runner (`tol=0`), PASS exige dlogits == 0
  (bit-exacto).
- Tensor arena: 2MB en PSRAM (`heap_caps_malloc` `MALLOC_CAP_SPIRAM`;
  `#define TFLM_ARENA_SIZE` en `main.cc`). El uso real se imprime al boot
  ("arena usada X de Y bytes").
- Target: devkit ESP32-S3 N16R8. Partición `factory` de 8MB
  (`partitions.csv`, flash 16MB), mismo layout que `firmware_rt/onnx2c`.
- Ops registradas (MicroMutableOpResolver): Conv2D, DepthwiseConv2D,
  FullyConnected, MaxPool2D, Mean (las que usan los 4 modelos del plan) +
  AveragePool2D, Reshape, Softmax, Pad, Transpose (defensivas).

## Las dos variantes de build: esp-nn ON/OFF

La clave del experimento. esp-tflite-micro delega los kernels int8 en
**esp-nn**; el knob para apagar el SIMD es el **Kconfig de esp-nn**
(choice `NN_OPTIMIZATIONS` en su `Kconfig.projbuild`):

- `CONFIG_NN_OPTIMIZED=y` (default) → assembly SIMD del S3
  (`esp_nn_*_esp32s3`).
- `CONFIG_NN_ANSI_C=y` → implementaciones C portables (`esp_nn_*_ansi`);
  `esp_nn.h` resuelve los alias `esp_nn_*` por `#if defined(CONFIG_NN_OPTIMIZED)`.

`sdkconfig.ref` contiene sólo `CONFIG_NN_ANSI_C=y`. Comandos exactos:

```bash
cd firmware_rt/tflm
. ~/esp/esp-idf/export.sh

# variante optimizada (esp-nn SIMD) → build_espnn/
idf.py -B build_espnn -DSDKCONFIG=build_espnn/sdkconfig \
       -DSDKCONFIG_DEFAULTS="sdkconfig.defaults" build

# variante referencia (ANSI C) → build_ref/
idf.py -B build_ref -DSDKCONFIG=build_ref/sdkconfig \
       -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.ref" build

# flasheo (puerto USB-UART CH343):
idf.py -B build_espnn -p /dev/ttyUSB0 flash monitor
```

**Verificación de que build_ref/ NO linkea el SIMD** (hacerla tras cambiar
de versión del componente — es fácil creer que está apagado y que no lo esté):

```bash
nm build_ref/tflm_rt.elf   | grep -c ' esp_nn_.*esp32s3'   # debe dar 0
nm build_espnn/tflm_rt.elf | grep -c ' esp_nn_.*esp32s3'   # >0 (30 con 1.2.5)
```

Verificado 2026-08-14 (esp-tflite-micro 1.4.0, esp-nn 1.2.5): build_ref 0
símbolos `*_esp32s3` / 13 `*_ansi`; build_espnn 30 `*_esp32s3`.
(En build_espnn también aparecen algunos `*_ansi`: son el fallback del
dispatch de esp-nn para casos no cubiertos por el assembly — esperado.)

## Regenerar / cambiar de modelo

```bash
# 1. exportar (venv aislado de tools/rt_export/tflite/; genera
#    tools/rt_export/tflite/out/<source>/{*.tflite,model_data.h,golden_data.h,model_meta.json}):
uv run --project tools/rt_export/tflite python tools/rt_export/tflite/export_tflite.py --source <source>

# 2. instalar en este proyecto (copia a main/, gitignoreado):
firmware_rt/tflm/tools/install_model.sh <source>

# 3. build de la(s) variante(s) — comandos de arriba.
```

## Estado (2026-08-14)

Los 4 modelos del plan compilan; `build_espnn/` y `build_ref/` quedan con
`depthwise_rgb32_pruned_activation_30_ft2` listos para flashear.
Tamaños de `tflm_rt.bin` (variante espnn; ref es ~23KB más chico):

| modelo | bin (espnn) |
|---|---|
| depthwise_rgb32_pruned_activation_30_ft2 | 395,456 B |
| depthwise_rgb32_pruned_activation_50_ft10 | 391,024 B |
| standard_rgb32_pruned_activation_30_ft10 | 430,896 B |
| standard_rgb96_pruned_activation_20_ft10 | 691,856 B |

La arena usada por modelo la imprime el firmware al boot (requiere placa).

Bench: `script06-bench.py --runtime tflm --firmware-dir firmware_rt/tflm --build-dir build_ref`
(y `--runtime tflm_espnn` / `build_espnn`).

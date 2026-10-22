# firmware_rt/onnx2c — runtime onnx2c (C plano, float32) en ESP32-S3

Proyecto ESP-IDF mínimo (sin cámara) que corre el self-test de golden images
sobre un modelo convertido a C con [onnx2c](https://github.com/kraiskil/onnx2c).
Usa el componente compartido `firmware_rt/common/` (selftest_runner): imprime
`ready`, y por UART `t` corre el self-test / `c` una inferencia.

- Entrada: float32 **NCHW** ya normalizada (las golden vienen así en
  `golden_data.h`).
- Referencia: logits float de onnxruntime en PC sobre el mismo `model.onnx`.
  Comparación con tolerancia `ONNX2C_TOL = 0.01` (modo float del runner).
- Target: devkit ESP32-S3 N16R8. Partición `factory` de 8MB
  (`partitions.csv`, flash 16MB) para que entren los pesos float en .rodata.

## Regenerar / cambiar de modelo

```bash
# 1. (una vez) compilar onnx2c — clona el repo y, si el sistema no tiene
#    protobuf, compila uno estático local (~5 min):
tools/rt_export/onnx2c/build.sh          # desde examples/esp32-cam/

# 2. exportar el modelo (env principal del ejemplo; genera
#    tools/rt_export/onnx2c/out/<source>/{model.onnx,model.c,golden_data.h,model_meta.json}):
uv run python tools/rt_export/onnx2c/export_onnx2c.py --source <source>

# 3. instalar en este proyecto (copia a main/, gitignoreado):
firmware_rt/onnx2c/tools/install_model.sh <source>

# 4. build:
cd firmware_rt/onnx2c
. ~/esp/esp-idf/export.sh
idf.py set-target esp32s3   # sólo la primera vez
idf.py build
# flasheo: idf.py -p /dev/ttyUSB0 flash monitor   (puerto USB-UART CH343)
```

`install_model.sh` copia `model.c`/`golden_data.h`/`model_meta.json` y, si el
input es >= 64px, anota los buffers estáticos de `model.c` (`tu0`/`tu1`) con
`EXT_RAM_BSS_ATTR` para moverlos a PSRAM (no entran en DRAM interna; requiere
`CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY=y`, ya en `sdkconfig.defaults`).
Ojo: eso implica que la inferencia del modelo grande corre contra PSRAM.

## Estado (2026-08-14)

Los 4 modelos del plan convierten y buildean OK (tamaños de `onnx2c_rt.bin`):

| modelo | accuracy PC (onnxruntime, float) | bin |
|---|---|---|
| depthwise_rgb32_pruned_activation_30_ft2 | 31.50% | 373,936 B |
| depthwise_rgb32_pruned_activation_50_ft10 | 42.02% | 362,704 B |
| standard_rgb32_pruned_activation_30_ft10 | 56.49% | 529,168 B |
| standard_rgb96_pruned_activation_20_ft10 | 59.15% | 1,570,624 B (buffers en PSRAM) |

Bench: `script06-bench.py --runtime onnx2c --firmware-dir firmware_rt/onnx2c`.

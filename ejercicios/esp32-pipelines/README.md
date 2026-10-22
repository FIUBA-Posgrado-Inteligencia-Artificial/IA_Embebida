# ESP32 pipelines — Flujo completo de compresión y despliegue

Pipeline educativo del curso **CEIA IA Embebida**: diseño de una CNN → entrenamiento → pruning → cuantización → despliegue en placas ESP32 (AI-Thinker ESP32-CAM, devkits ESP32/ESP32-S3, ESP32-S3-CAM), con comparativa de runtimes de inferencia alternativos (`firmware_rt/`).

---

## Requisitos

- Python 3.10–3.11
- [UV](https://docs.astral.sh/uv/) como gestor de entorno
- Hardware (solo para las etapas de deploy): AI-Thinker ESP32-CAM + ESP32-CAM-MB, y/o devkit ESP32, devkit ESP32-S3, ESP32-S3-CAM (ver §"Firmware ESP-IDF")

```bash
cd examples/esp32-pipelines
uv sync           # crea .venv e instala dependencias
```

El dataset STL-10 (~2.6 GB) se descarga automáticamente en `examples/datasets/` la primera vez que se ejecuta cualquier script de entrenamiento.

---

## Verificación del hardware

Antes de correr el pipeline completo, conviene confirmar que el módulo ESP32-CAM y su programador MB están funcionando. El subproyecto [`firmware_test_webserver/`](firmware_test_webserver/) contiene un firmware ESP-IDF mínimo para eso (wifi + cámara + stream MJPEG, sin inferencia).

### Hardware confirmado

| Componente | Detalle |
|---|---|
| SoC | ESP32-D0WD-V3 rev 3.0, dual-core 240 MHz |
| Flash | 4 MB |
| PSRAM | 8 MB quad-SPI (detectada y testeada OK) |
| Sensor | OV2640 — PID 0x26, dirección I2C 0x30 |
| Programador | ESP32-CAM-MB con CH341 (aparece como `ttyUSB*`) |

### Procedimiento de flash (MB no tiene auto-reset)

El MB no activa el auto-reset via DTR/RTS. Hay que hacer un **cold boot** manual antes de cada flash:

```
1. Desenchufar el USB del MB
2. Mantener apretado [BOOT] (botón IO0 del MB)
3. Enchufar el USB
4. Soltar [BOOT]
```

Inmediatamente:

```bash
esptool.py --chip esp32 -p /dev/ttyUSB2 -b 115200 --before no_reset \
  write_flash --flash_mode dio --flash_freq 40m --flash_size 4MB \
  0x1000  build/bootloader/bootloader.bin \
  0x8000  build/partition_table/partition-table.bin \
  0x10000 build/esp32cam_test.bin
```

Cuando termina: presionar **[RST]** para bootear normal. El monitor se sale con `Ctrl+]`.

> `script05-build.py --action flash` usa `idf.py flash` que también falla sin el cold boot. Si se usa el script, hacer el cold boot antes de lanzarlo o usar `--no-preflight` + cold boot manual.

### Bug conocido: `ACK_CHECK_EN` en `sccb.c`

`espressif/esp32-camera ^2.0.0` usa `ACK_CHECK_EN` en todas las escrituras SCCB. El protocolo SCCB de OmniVision define el 9.º bit como "don't care" (no es un ACK real), por lo que el driver I2C de ESP-IDF v5.x falla la transacción.

**Síntomas:**

| Error en monitor | Causa |
|---|---|
| `SCCB_Write Failed ret:263` | XCLK demasiado rápido (timeout) |
| `SCCB_Write Failed ret:-1` | ACK check falla + contención DMA+PSRAM |

**Workaround aplicado** (tanto en el test como en el firmware):
- `xclk_freq_hz = 8 000 000` (en lugar de 20 MHz)
- `fb_location = CAMERA_FB_IN_DRAM` durante la inicialización

Si en el futuro se necesita `CAMERA_FB_IN_PSRAM` para frames grandes (UXGA), la solución limpia es cambiar `ACK_CHECK_EN` → `ACK_CHECK_DIS` en las escrituras de `managed_components/espressif__esp32-camera/driver/sccb.c`.

### Pinout AI-Thinker ESP32-CAM

```
PWDN  → GPIO 32    XCLK  → GPIO  0
RESET → N/C        SIOD  → GPIO 26  (I2C SDA)
VSYNC → GPIO 25    SIOC  → GPIO 27  (I2C SCL)
HREF  → GPIO 23    PCLK  → GPIO 22
Y9    → GPIO 35    Y8    → GPIO 34
Y7    → GPIO 39    Y6    → GPIO 36
Y5    → GPIO 21    Y4    → GPIO 19
Y3    → GPIO 18    Y2    → GPIO  5
```

---

## Estructura

```
esp32-pipelines/
├── main.py                         # Entry point: ejecuta flows declarados en pipelines.yaml
├── pipelines.yaml                  # Flows (train→…→monitor) por modelo y placa
├── pipelib/                        # Utilidades importables
│   ├── pipeline_runner.py          # Carga/valida YAML, stage → argv, ejecución fail-fast
│   ├── model_registry.py           # Registry de modelos (models.json) y benchmarks (benchmarks.json)
│   ├── esp32cam_utils.py           # Utilidades compartidas (modelos, training, pruning, tabla)
│   └── quantize_utils.py           # Infra común a los tres scripts de cuantización
├── scripts/                        # Scripts de pipeline y generadores
│   ├── script01-train.py           # Entrenamiento (todas las variantes CNN)
│   ├── script02-prune.py           # Pruning estructurado (weight / activation), sin fine-tuning
│   ├── script03-finetune.py        # Fine-tuning explícito post-pruning
│   ├── script04a-quantize.py       # Cuantización PTQ (esp-ppq, sin QAT real)
│   ├── script04b-quantize-torchao.py  # QAT real con torchao + PTQ final con esp-ppq
│   ├── script04c-quantize-brevitas.py # QAT real con brevitas, puentes preconditioned/qdq
│   ├── script05-build.py           # build / flash / monitor del firmware (--board, --action)
│   ├── script06-bench.py           # Benchmark on-device vía self-test (registra en benchmarks.json)
│   ├── gen_static_frame.py         # Imagen fija embebida para placas sin cámara
│   └── gen_test_images.py          # Golden images standalone (experimentos; el flujo real usa 04a)
├── models.json / benchmarks.json   # Registry de modelos / mediciones on-device (trackeados)
├── firmware/                       # Proyecto ESP-IDF unificado multi-placa (esp32/esp32cam/esp32s3/esp32s3cam)
├── firmware_rt/                    # Runtimes alternativos: common/ (self-test), tflm/, onnx2c/, executorch/
├── firmware_test_webserver/        # Isolation test: wifi + cámara + MJPEG, sin inferencia
├── tools/
│   ├── build_matrix.sh             # Smoke test: compila las 4 placas de punta a punta
│   └── rt_export/                  # Exports por runtime (tflite/, onnx2c/, executorch/), venvs aislados
├── models/                         # Checkpoints .pth y exportaciones .espdl
└── tests/                          # pytest: utils, quantize, registry, CLI, build, bench (198 tests)
```

El paso de cuantización tiene tres variantes (a/b/c) que producen todas un
`.espdl` consumible por el firmware — eligés cuál según el método de
cuantización que quieras demostrar. Todas comparten el flujo
01 → 02 → 03 → ... anterior.

---

## Flujo de trabajo

```
script01-train → script02-prune → script03-finetune → script04{a|b|c}-quantize* → script05-build (build/flash/monitor) → script06-bench
(entrena)        (prune, sin FT)   (fine-tuning)        (cuantiza, tres variantes)  (despliega)                           (mide on-device, opcional)
```

`prune` ya no hace fine-tuning internamente — es un stage separado
(`finetune`) que el usuario decide correr o no. No hay validación de
dependencias entre stages: si renombrás el output de un script, hay que
actualizar a mano el `--source`/`source:` del siguiente.

Cada script guarda el checkpoint con un nombre que concatena los pasos aplicados:

```
standard_rgb96.pth
depthwise_gray32.pth
depthwise_gray32_pruned_activation_30.pth
depthwise_gray32_pruned_activation_30_ft10.pth                                  # 03-finetune, 10 epochs
depthwise_gray32_pruned_activation_30_ft10_quantized_ptq.espdl                  # 04a
depthwise_gray32_pruned_activation_30_ft10_quantized_qat_torchao.espdl          # 04b
depthwise_gray32_pruned_activation_30_ft10_quantized_qat_brevitas.espdl         # 04c preconditioned
depthwise_gray32_pruned_activation_30_ft10_quantized_qat_brevitas_qdq.espdl     # 04c qdq
```

Si el checkpoint de salida ya existe, el script lo carga y omite el procesamiento.

---

## Script de entrenamiento (01-train)

Dataset: **STL-10** — 10 clases, 5000 train / 8000 test, 96×96 RGB.

```bash
# Las 5 variantes del flujo educativo:
uv run python scripts/script01-train.py --arch standard  --input-mode rgb  --resolution 96  # baseline
uv run python scripts/script01-train.py --arch standard  --input-mode rgb  --resolution 32  # menor resolución
uv run python scripts/script01-train.py --arch standard  --input-mode gray --resolution 32  # escala de grises
uv run python scripts/script01-train.py --arch depthwise --input-mode rgb  --resolution 32  # depthwise
uv run python scripts/script01-train.py --arch depthwise --input-mode gray --resolution 32  # candidato a pruning
```

| Variante | Arquitectura | Entrada | Checkpoint generado |
|----------|-------------|---------|---------------------|
| `--arch standard  --input-mode rgb  --resolution 96` | CNN estándar | RGB 96×96 | `standard_rgb96.pth` |
| `--arch standard  --input-mode rgb  --resolution 32` | CNN estándar | RGB 32×32 | `standard_rgb32.pth` |
| `--arch standard  --input-mode gray --resolution 32` | CNN estándar | gray 32×32 | `standard_gray32.pth` |
| `--arch depthwise --input-mode rgb  --resolution 32` | CNN depthwise | RGB 32×32 | `depthwise_rgb32.pth` |
| `--arch depthwise --input-mode gray --resolution 32` | CNN depthwise | gray 32×32 | `depthwise_gray32.pth` |

| Flag | Valores | Default |
|------|---------|---------|
| `--arch` | `standard` \| `depthwise` | requerido |
| `--input-mode` | `rgb` \| `gray` | requerido |
| `--resolution` | entero positivo (STL-10 nativo: 96) | requerido |
| `--epochs` | entero | `30` |
| `--lr` | float | `1e-3` |
| `--batch-size` | entero | `64` |
| `--force` | — | reentrenar aunque el checkpoint exista |

Al finalizar se imprime una tabla comparativa con todos los modelos entrenados hasta ese momento (accuracy, parámetros, MACs, KB).

### Arquitecturas

**StandardCNN** — 3 bloques `Conv(3×3) → BN → ReLU → MaxPool`, seguidos de `AdaptiveAvgPool(1×1) → Linear(10)`.

**DepthwiseCNN** — misma topología pero cada bloque usa convolución depthwise separable (`DW 3×3 + PW 1×1`). Reduce los MACs ~8-9× respecto al estándar.

Ambas arquitecturas son agnósticas a la resolución gracias a `AdaptiveAvgPool`.

---

## Script de pruning (02-prune)

`script02-prune.py` **ya no hace fine-tuning** — sólo poda y guarda el
checkpoint con la topología reducida. El fine-tuning es un stage separado
(`script03-finetune.py`, ver abajo).

```bash
# Pruning por activación, 30% de canales eliminados
uv run python scripts/script02-prune.py --source depthwise_gray32 --method activation --ratio 0.3

# Pruning por magnitud de pesos, 50%
uv run python scripts/script02-prune.py --source standard_rgb32 --method weight --ratio 0.5

# Sin pruning (baseline para comparar)
uv run python scripts/script02-prune.py --source depthwise_gray32 --method none
```

| Flag | Valores | Default |
|------|---------|---------|
| `--source` | nombre del checkpoint (sin `.pth`) | requerido |
| `--method` | `none` \| `weight` \| `activation` | `activation` |
| `--ratio` | fracción de canales a eliminar | `0.3` |
| `--batch-size` | tamaño de lote | `64` |

El checkpoint resultante guarda la topología reducida (número real de canales supervivientes) para que `script03-finetune.py`/`script04*-quantize.py` puedan reconstruir el modelo correctamente.

---

## Script de fine-tuning (03-finetune)

Stage explícito, corrido a discreción del usuario después de `prune` (o
directamente sobre un baseline, sin podar). Guarda el checkpoint como
`<source>_ft<epochs>.pth`.

```bash
uv run python scripts/script03-finetune.py --source depthwise_gray32_pruned_activation_30 --epochs 10 --lr 2e-4
```

| Flag | Valores | Default |
|------|---------|---------|
| `--source` | nombre del checkpoint (sin `.pth`) | requerido |
| `--epochs` | epochs de fine-tuning | `10` |
| `--lr` | learning rate | `2e-4` |
| `--batch-size` | tamaño de lote | `64` |

---

## Scripts de cuantización (04a / 04b / 04c)

El paso 3 del flujo ofrece tres caminos alternativos al `.espdl`. Los tres
generan los mismos artefactos (`models/{nombre}_quantized_*.espdl` +
`firmware/main/model_data.h`) y se distinguen por cómo (o si) aplican QAT.

Target: **ESP32 clásico** — esquema de cuantización int8 simétrico, per-tensor,
con scales potencia de 2. Todos los flujos terminan en `esp-ppq` respetando
esa restricción.

> **Nota:** `esppq` no está en PyPI. Instalar desde el repositorio oficial:
> ```bash
> uv add "esppq @ git+https://github.com/espressif/esp-ppq.git"
> ```
> Sin `esppq`, los scripts exportan `.pth` de fallback y generan `model_data.h`
> con un array vacío (útil para probar el flujo sin el toolchain completo).

### 04a — esp-ppq directo (baseline, sin QAT real)

```bash
# PTQ puro
uv run python scripts/script04a-quantize.py --source depthwise_gray32_pruned_activation_30 --mode ptq

# "QAT" = fine-tune sobre el modelo ya PTQ-cuantizado (no es QAT con fake-quant
# real durante training; para eso ver 04b/04c)
uv run python scripts/script04a-quantize.py --source depthwise_gray32 --mode qat
```

| Flag | Valores | Default |
|------|---------|---------|
| `--source` | nombre del checkpoint (sin `.pth`) | requerido |
| `--mode` | `ptq` \| `qat` | `ptq` |
| `--calib-batches` | batches de calibración PTQ | `32` |
| `--qat-epochs` | epochs de fine-tune post-PTQ | `5` |
| `--firmware-dir` | destino de `model_data.h` | `firmware/main/` |

### 04b — torchao QAT + esp-ppq PTQ

Inserta fake-quant **real** durante training usando el flujo eager-mode de
`torch.ao.quantization`, entrena contra esos nodos, strip-ea los fake-quant
y pasa el `.pth` resultante por el mismo `esp-ppq` que usa 04a. Los scales
aprendidos se descartan (los re-calibra esp-ppq); lo que sobrevive es la
robustez al ruido de cuantización.

```bash
uv run python scripts/script04b-quantize-torchao.py --source depthwise_gray32_pruned_activation_30
uv run python scripts/script04b-quantize-torchao.py --source standard_rgb32 --qat-epochs 10
```

| Flag | Valores | Default |
|------|---------|---------|
| `--source` | nombre del checkpoint (sin `.pth`) | requerido |
| `--qat-epochs` | epochs de QAT con fake-quant | `5` |
| `--qat-lr` | learning rate del QAT | `1e-4` |
| `--calib-batches` | batches de calibración PTQ post-strip | `32` |
| `--firmware-dir` | destino de `model_data.h` | `firmware/main/` |

### 04c — brevitas QAT con dos puentes

Usa brevitas (QAT con fake-quant basados en `QuantConv2d`/`QuantLinear`).
Ofrece dos puentes hacia `.espdl`, seleccionables con `--bridge`:

- **`preconditioned`** (default): igual patrón que 04b — QAT con scales float,
  strip, esp-ppq PTQ. Los scales aprendidos se descartan.
- **`qdq`**: QAT con scales **potencia de 2** (coincide con el esquema de
  ESP-DL por construcción), exporta ONNX con nodos QDQ cargando las escalas
  aprendidas, y los pasa a `esppq.espdl_quantize_onnx(target="c")`. Las
  escalas aprendidas en QAT sobreviven end-to-end hasta el `.espdl`.

```bash
# Puente A (preconditioned) — default
uv run python scripts/script04c-quantize-brevitas.py --source depthwise_gray32_pruned_activation_30

# Puente B (qdq) — end-to-end con PoT scales
uv run python scripts/script04c-quantize-brevitas.py --source depthwise_gray32_pruned_activation_30 --bridge qdq
```

| Flag | Valores | Default |
|------|---------|---------|
| `--source` | nombre del checkpoint (sin `.pth`) | requerido |
| `--bridge` | `preconditioned` \| `qdq` | `preconditioned` |
| `--qat-epochs` | epochs de QAT | `5` |
| `--qat-lr` | learning rate del QAT | `1e-4` |
| `--calib-batches` | batches de calibración | `32` |
| `--firmware-dir` | destino de `model_data.h` | `firmware/main/` |

### ¿Cuál usar?

| Si querés... | Script |
|---|---|
| Ver el baseline sin QAT, comparar contra los otros | **04a** |
| Mostrar QAT con la API nativa de PyTorch (torchao) | **04b** |
| Mostrar QAT con scales PoT (el esquema real de ESP-DL) | **04c --bridge qdq** |
| Un QAT portable que funciona incluso si el qdq bridge falla | **04c --bridge preconditioned** |

Artefactos comunes a los tres scripts:

- **`models/{nombre}_quantized_*.espdl`** — modelo en formato ESP-DL.
- **`firmware/main/model_data.h`** — header C con el modelo como array
  `uint8_t` y las constantes de preprocesamiento:

```c
#define MODEL_INPUT_W          32
#define MODEL_INPUT_H          32
#define MODEL_INPUT_CHANNELS   1   /* 1=gray, 3=RGB */
#define MODEL_INPUT_MEAN_0     112
#define MODEL_INPUT_STD_0      56
```

---

## Tests

```bash
# Todo el suite
uv run pytest tests/ -v

# Excluir los smoke tests de QAT (que entrenan un paso)
uv run pytest tests/ -v -m "not slow"
```

Cubre:
- `test_utils.py` — forward shapes de todas las combinaciones arq/modo/resolución, entropía normalizada (uniforme=1, one-hot=0), checkpoint roundtrip, y reducción de canales tras pruning.
- `test_quantize.py` — `generate_model_data_h` (gray/RGB/sin espdl), smoke tests marcados `slow` para el strip de torchao, el roundtrip FP32↔twin brevitas, el puente preconditioned y el export ONNX QDQ.

---

## Métricas de los modelos

Valores medidos de los modelos desplegados (fuente de verdad:
`uv run main.py models`, que agrega los tiempos on-device de `benchmarks.json`):

| Modelo (cuantizado int8) | Acc | Params | MACs | espdl esp32cam | espdl esp32s3 |
|--------------------------|-----|--------|------|----------------|---------------|
| depthwise rgb32, prune 50% + ft10 | 42.0% | 4.2 K | 395 K | 75.7 ms | 5.1 ms |
| standard rgb32, prune 30% + ft10 | 56.5% | 45.9 K | 5.1 M | 665 ms | 48.6 ms |
| standard rgb96 (good_model), prune 20% + ft10 | 59.2% | 60.5 K | 59.6 M | 9366 ms | 619 ms (PSRAM) |

---

## Entry point: `main.py`

`main.py` orquesta el pipeline completo (train → prune → finetune → quantize → build → flash → monitor) a partir de flows declarados en `pipelines.yaml`. Cada flow es una lista ordenada de stages self-contained (todos los parámetros declarados explícitamente), o un `extends: <flow>` + `stages: [...]` que hereda los stages de otro flow y agrega los propios — como los scripts cachean sus outputs, re-correr lo heredado es casi gratis (ej.: `good_model_esp32` extiende `good_model`, que extiende `good_model_base`; `good_model_esp32s3` reusa la misma base con quantize/build del S3). Los flows genéricos `flash` y `monitor` no dependen de ningún modelo.

### Subcomandos

```bash
uv run python main.py run <flow>           # ejecuta un flow
uv run python main.py run <flow> --dry-run # sólo imprime los comandos que correría
uv run python main.py flows                # lista flows definidos
uv run python main.py models               # imprime la tabla de modelos en models/
uv run python main.py models clean         # reconcilia models.json con el disco
uv run python main.py models clean --dry-run  # muestra cambios sin escribir
```

### Flows provistos

Los flows de referencia viven en `pipelines.yaml`. Entre otros:

| Flow | Stages | Uso |
|------|--------|-----|
| `flash` / `monitor` | flash / monitor (`board: esp32cam`) | genéricos: flashean lo último buildeado / abren el monitor, sin importar el modelo. Usan `board: esp32cam`; tras buildear otra placa, flashear con `uv run python scripts/script05-build.py --action flash --board <board>` |
| `fast_debug` (`_base`) | train → prune → finetune → quantize | ciclo corto (minutos) para verificar el pipeline sin placa |
| `fast_debug_esp32` | extends `fast_debug` + build → flash → monitor (`board: esp32cam`) | deploy completo del ciclo corto en el ESP32-CAM |
| `benchmark_esp32` | extends `fast_debug` + build → flash → monitor (`board: esp32`) | mide preprocess+inferencia en ESP32 clásico sin cámara (imagen fija embebida) |
| `benchmark_esp32s3` | extends `fast_debug_base` + quantize (`target: esp32s3`) → build (`board: esp32s3`) | build-only en ESP32-S3 sin cámara; sin placa para flashear |
| `fast_debug_esp32s3cam` | extends `fast_debug_base` + quantize (`target: esp32s3`) → build (`board: esp32s3cam`) | build-only para la ESP32-S3-CAM (OV3660), proyecto único `firmware/` |
| `good_model` (`_base`) / `good_model_esp32` | hasta quantize / + build → flash → monitor (`board: esp32cam`) | standard rgb 96×96, mayor calidad |
| `standard32` (`_base`) | train (30 ep) → prune 30% → ft10 → quantize | standard rgb 32×32, punto medio calidad/tamaño |
| `small_model` (`_base`) / `small_model_esp32` | hasta quantize / + build → flash → monitor (`board: esp32cam`) | depthwise rgb 32×32 (30 ep, prune 50%), mayor compresión |

Las placas `esp32`/`esp32s3`/`esp32s3cam` corren en el **mismo** proyecto
`firmware/` (multi-placa, sin cámara en `esp32`/`esp32s3`): no hay flows
separados por proyecto S3 externo.

Definir un nuevo flow es editar `pipelines.yaml` — no requiere tocar Python.

### `pipelines.yaml`

Cada entrada del flow es un diccionario con:

- `stage`: `train` | `prune` | `finetune` | `quantize` | `build` | `flash` | `monitor`
- `source`: tag del modelo de entrada (requerido para `prune`, `finetune` y `quantize`)
- `variant`: `a` | `b` | `c` (sólo para `quantize`; default `a`)
- `params`: dict de flags que se traducen a `--kebab-case`

No hay validación cruzada entre stages: si el output de un stage no coincide
con el `source:` del siguiente, el error aparece recién al correrlo.

Un `port` a nivel top-level del YAML se inyecta automáticamente en
`params.port` de los stages `flash`/`monitor` que no lo definan (no en
`build`).

Ver el YAML incluido en este directorio para los flows de referencia completos.

### `script05-build.py`

Implementa los stages `build`, `flash` y `monitor` vía la flag `--action`
(default `build`). `build` compila con preflight y no requiere puerto;
`flash` y `monitor` resuelven el puerto serie. Se puede correr standalone:

```bash
cd examples/esp32-pipelines
uv run python scripts/script05-build.py --action build              # sólo compila (preflight + idf.py build)
uv run python scripts/script05-build.py --action flash --port /dev/ttyUSB0
uv run python scripts/script05-build.py --action monitor --port /dev/ttyUSB0
uv run python scripts/script05-build.py --action build --no-preflight  # salta los chequeos previos
```

Preflight chequea: `IDF_PATH`, `idf.py` en PATH, `firmware/main/model_data.h` presente, target instalado coherente con `--target`, coherencia entre `model_data.h` y `model_data.meta.json`, PSRAM habilitada, y (para `flash`/`monitor`) puerto USB accesible.

### `script06-bench.py`

Mide el tiempo de inferencia **en la placa** y lo registra en
`benchmarks.json` (key: modelo + placa + runtime). Flashea el build indicado,
espera el `ready` por serie, dispara el self-test (tecla `t`), parsea las
líneas de resultado (tiempos + paridad de logits contra la referencia de PC)
y guarda media/min/max con el conteo de PASS:

```bash
cd examples/esp32-pipelines
# Camino espdl clásico (usa firmware/build_<board>/)
uv run python scripts/script06-bench.py --source depthwise_rgb32_pruned_activation_50_ft10_quantized_ptq \
    --board esp32s3 --port /dev/ttyUSB0

# Runtime alternativo: apuntar al build dir del proyecto firmware_rt correspondiente
uv run python scripts/script06-bench.py --source <modelo>_quantized_ptq --board esp32s3 --port /dev/ttyUSB0 \
    --runtime tflm_espnn --build-dir firmware_rt/tflm/build_espnn --skip-build
```

Flags: `--runtime {espdl,tflm,tflm_espnn,onnx2c,executorch}` (default
`espdl`), `--build-dir` (override del build a flashear), `--skip-build` /
`--skip-flash` para reutilizar lo ya compilado/flasheado. Las mediciones
se ven con `uv run main.py bench` (una fila por modelo/placa/runtime); `main.py models` marca en la columna `Bench` qué placas tienen datos.

Los flows `benchmark_esp32` / `benchmark_esp32s3` de `pipelines.yaml`
encadenan build + bench para el camino espdl.

---

## Firmware ESP-IDF

`firmware/` es un **proyecto único multi-placa** (`esp32`, `esp32cam`,
`esp32s3`, `esp32s3cam`, seleccionable con `idf.py -DBOARD=<board>`, default
`esp32cam`) que consume `model_data.h` generado por
`script04{a,b,c}-quantize*.py`. La configuración vive en dos headers
trackeados, no en Kconfig: `firmware/main/boards/board_<nombre>.h` (lo
intrínseco de cada placa: cámara, PSRAM, pines) y
`firmware/main/firmware_config.h` (features de usuario: WiFi, webserver,
self-test, PSRAM override, etc.) — ver `CLAUDE.md` § "Placas y
configuración" para el detalle completo y las reglas de coherencia
(`FEATURE_WEBSERVER` sin `FEATURE_WIFI` no compila). Las placas sin cámara
(`esp32`, `esp32s3`) corren el mismo pipeline de preprocess+inferencia sobre
una imagen fija embebida, útil para benchmarking sin módulo de cámara.

Soporta:

- **Estado compartido de captura** (`capture_state.{c,h}`): último resultado de inferencia, último JPEG y modo activo, consumidos tanto por el trigger serie como por el servidor HTTP.
- **Dos modos**: `single` (una captura por trigger) y `continuous` (loop libre con `fps`). Se togglea con la tecla `m` por serie o el checkbox de la página web.
- Trigger por serie: menú de teclas — `c` captura+inferencia (en modo single, con progreso por etapas), `t` self-test, `m` toggle de modo, `h`/`?` ayuda; el resto se ignora.
- **Self-test en hardware** (tecla `t`, deshabilitable con `FEATURE_HW_VALIDATION 0`): corre la inferencia sobre golden images del test set de STL-10 embebidas en `firmware/main/test_images.h` e imprime device vs PC con PASS/FAIL por clase. El header lo emite `script04a-quantize.py` con el mismo grafo cuantizado que exporta el `.espdl` (referencia exacta) y lo instala `script05-build.py` junto al modelo; el PASS esperable es N/N estricto y un FAIL indica un bug real en la inferencia embebida. Si pasa, cualquier resultado raro con la cámara es de captura/preprocess/dominio o calidad del modelo cuantizado.
- Código de trigger/publicación por MQTT presente pero compilado out por defecto (fuera de la config nueva; ver `esp32cam_mqtt.c`).
- Servidor HTTP (`FEATURE_WEBSERVER`, requiere `FEATURE_WIFI`) con página de control y endpoints: `GET /` (página con stream, trigger y checkbox de modo), `GET /stream` (MJPEG), `POST /trigger`, `POST /mode` (`continuous=0|1`), `GET /last.jpg`, `GET /result` (JSON polleado cada 500 ms con `seq`, `mode`, `last_error`, `top1`, `probs`, `class_names`, `t_capture_ms`, `t_preprocess_ms`, `t_inference_ms`, `fps`).
- Salida dual: texto humano + JSON en serie.

Build rápido (una placa específica):

```bash
cd examples/esp32-pipelines/firmware
idf.py -B build_esp32cam -DBOARD=esp32cam set-target esp32
idf.py -B build_esp32cam -DBOARD=esp32cam build
idf.py -B build_esp32cam -p /dev/ttyUSB0 flash monitor
```

o, preferido, vía `script05-build.py --board <board>` (preflight incluido) o
un flow de `pipelines.yaml`.

Matriz de builds (smoke test de las 4 placas):

```bash
cd examples/esp32-pipelines
. ~/esp/esp-idf/export.sh
tools/build_matrix.sh
```

Ver [`firmware/README.md`](./firmware/README.md) para el procedimiento completo, `menuconfig` de config IDF pura (particiones, log level — ya no features del firmware) y tests de host del módulo de preprocesamiento.

---

## Runtimes alternativos (`firmware_rt/`)

Además del camino principal esp-ppq → ESP-DL, el directorio `firmware_rt/`
contiene proyectos ESP-IDF mínimos (ESP32-S3 devkit, sin cámara) que corren
el **mismo self-test de golden images** sobre otros cuatro runtimes de
inferencia, para comparar en igualdad de condiciones:

| Runtime | Precisión | Kernels | Proyecto |
|---|---|---|---|
| `espdl` | int8 | ESP-DL (SIMD S3) | `firmware/` (board `esp32s3`) |
| `tflm_espnn` | int8 | TFLite Micro + esp-nn (SIMD S3) | `firmware_rt/tflm/` → `build_espnn/` |
| `tflm` | int8 | TFLite Micro referencia (C plano, `CONFIG_NN_ANSI_C`) | `firmware_rt/tflm/` → `build_ref/` |
| `onnx2c` | float | C plano generado por onnx2c | `firmware_rt/onnx2c/` |
| `executorch` | float | ExecuTorch portable ops | `firmware_rt/executorch/` |

### Resultados (ESP32-S3 devkit N16R8, ms por inferencia, self-test PASS en todos)

| modelo | espdl int8 | tflm_espnn int8 | tflm ref int8 | onnx2c float | executorch float |
|---|---|---|---|---|---|
| `depthwise_rgb32_p30_ft2` | 11.2 | 35 | 66 | 25.1 | 851.6 |
| `depthwise_rgb32_p50_ft10` | 5.1 | 16.2 | 45 | 16.5 | 516.4 |
| `standard_rgb32_p30_ft10` | 48.6 | 72 | 248 | 462 | 4503.3 |
| `standard_rgb96_p20_ft10` | 619.4 | 797 | 3545 | 7047.8 | 56585.7 ¹ |

¹ bench cortado a 3 imágenes por timeout; los tiempos por imagen son consistentes.

**Análisis:**

- **int8 + SIMD gana siempre**: `espdl` es el más rápido en los 4 modelos.
- **esp-nn acelera TFLM 1.9–4.4×** vs los kernels de referencia — más cuanto
  más convolución densa tiene el modelo.
- **ESP-DL le gana a TFLM+esp-nn 1.3–3×** aun siendo ambos int8+SIMD.
- **C float plano (onnx2c) compite con TFLM int8 de referencia** en los
  modelos depthwise chicos, pero pierde ~10× en las conv densas.
- **ExecuTorch portable (float) es 30–90× más lento que espdl** — el costo de
  kernels genéricos sin cuantización delegada en Xtensa (hoy no existe
  backend delegado para el S3).

**Paridad de resultados:** los runtimes int8 comparan logits int8 contra la
misma toolchain en PC — `espdl` es bit-exacto (referencia del mismo grafo
esp-ppq); TFLM admite `dlogits<=1` (1 LSB) contra el interpreter LiteRT de
x86, porque el redondeo del requant difiere entre implementaciones. Los
runtimes float (onnx2c, executorch) comparan contra onnxruntime / el runtime
ET de PC con tolerancia 0.01 y dan `dlogits=0`.

### Cómo regenerar y medir un runtime

El patrón es siempre: **export en PC → install → build → bench**. Ejemplo con
`depthwise_rgb32_pruned_activation_30_ft2`:

```bash
# 1. Export (cada runtime tiene su venv aislado bajo tools/rt_export/<rt>/)
uv run --project tools/rt_export/tflite     python tools/rt_export/tflite/export_tflite.py         --source depthwise_rgb32_pruned_activation_30_ft2
uv run --project tools/rt_export/executorch python tools/rt_export/executorch/export_executorch.py --source depthwise_rgb32_pruned_activation_30_ft2
python tools/rt_export/onnx2c/export_onnx2c.py --source depthwise_rgb32_pruned_activation_30_ft2

# 2. Instalar los headers generados en el proyecto firmware
firmware_rt/tflm/tools/install_model.sh          depthwise_rgb32_pruned_activation_30_ft2
firmware_rt/onnx2c/tools/install_model.sh        depthwise_rgb32_pruned_activation_30_ft2
firmware_rt/executorch/install_model.sh          depthwise_rgb32_pruned_activation_30_ft2

# 3. Build (ESP-IDF; TFLM tiene DOS builds: SIMD y referencia)
. ~/esp/esp-idf/export.sh
(cd firmware_rt/tflm && idf.py -B build_espnn -DSDKCONFIG=build_espnn/sdkconfig -DSDKCONFIG_DEFAULTS="sdkconfig.defaults" build)
(cd firmware_rt/tflm && idf.py -B build_ref   -DSDKCONFIG=build_ref/sdkconfig   -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.ref" build)
(cd firmware_rt/onnx2c && idf.py build)
(cd firmware_rt/executorch && idf.py build)

# 4. Bench (flashea, espera "ready", manda 't' y parsea el self-test)
uv run python scripts/script06-bench.py --source depthwise_rgb32_pruned_activation_30_ft2 --board esp32s3 \
    --runtime tflm_espnn --build-dir firmware_rt/tflm/build_espnn --skip-build
uv run python scripts/script06-bench.py --source ... --runtime tflm       --build-dir firmware_rt/tflm/build_ref  --skip-build
uv run python scripts/script06-bench.py --source ... --runtime onnx2c     --build-dir firmware_rt/onnx2c/build     --skip-build
uv run python scripts/script06-bench.py --source ... --runtime executorch --build-dir firmware_rt/executorch/build --skip-build
```

Los resultados se acumulan en `benchmarks.json` y `uv run python main.py models`
los muestra `main.py bench` como filas por runtime.

Detalles por runtime en los README de cada proyecto
(`firmware_rt/{tflm,onnx2c,executorch}/README.md`). Gotchas destacados:

- **TFLM**: la variante de referencia se logra con `CONFIG_NN_ANSI_C=y`
  (`sdkconfig.ref`); verificar con `nm` que `build_ref` no linkea símbolos
  `esp_nn_*_esp32s3`.
- **onnx2c rgb96**: los buffers intermedios no entran en RAM interna — el
  `install_model.sh` los anota con `EXT_RAM_BSS_ATTR` para mandarlos a PSRAM.
- **ExecuTorch v1.4.0** trae soporte ESP-IDF upstream (preset
  `esp-baremetal`). Dos trampas: el gcc de Xtensa hace ICE con `-O2` en
  `op_sum.cpp` (usar `EXECUTORCH_OPTIMIZE_SIZE=ON`), y **no** duplicar el
  `--whole-archive` de `portable_ops_lib` (ya viene en
  `INTERFACE_LINK_OPTIONS`; duplicado = doble registro de kernels y crash al
  boot).

---

## Efecto de la memoria en la latencia (experimentos, 2026-08-20)

Dos experimentos sobre la devkit ESP32-S3 (N16R8) y la ESP32-S3-CAM, con los
`.espdl` target esp32s3 y el self-test como carga (medición manual, no
registrada en `benchmarks.json`). El firmware expone `MODEL_PLACEMENT` en
`firmware_config.h` (`AUTO`/`FLASH`/`PSRAM`/`INTERNAL`) para forzar dónde
viven los **pesos** del modelo: copia el blob a la memoria pedida y esp-dl
lee de ahí (`param_copy` off).

### 1. Placement de pesos: flash vs PSRAM vs SRAM interna (devkit)

| Modelo (pesos) | FLASH | PSRAM | INTERNAL |
|---|---|---|---|
| `depthwise_..._30_ft2` (18 KB) | 11.4 ms | 11.5 ms | 12.2 ms |
| `standard_rgb32_..._ft10` (52 KB) | 60.4 ms | 37.8 ms | 37.7 ms |

Lecturas:

- **Si el working set entra en la caché, la jerarquía es invisible**: el
  modelo de 18 KB corre igual desde cualquier memoria — tras la primera
  pasada todos los accesos pegan en la caché del S3.
- **Cuando desborda la caché, la que pierde es la flash** (+60%): cada miss
  va al SPI de la flash, que además compite con el fetch de instrucciones
  (el código también corre desde flash por la misma caché).
- **PSRAM octal ≈ SRAM interna** para este patrón de acceso: el streaming
  secuencial de pesos a través de la caché absorbe la latencia extra de la
  PSRAM.

### 2. Por qué la S3-CAM es ~3× más lenta que la devkit (mismo chip)

Con `standard_rgb32` (39.1 ms en la devkit, 125.8 ms en la S3-CAM):

| Config | t_inf |
|---|---|
| devkit, PSRAM activa, sin cámara (baseline) | 39.3 ms |
| devkit, `FEATURE_PSRAM 0`, sin cámara | 59.8 ms |
| S3-CAM (PSRAM rota), firmware **sin** cámara (board `esp32s3`) | 53.3 ms |
| S3-CAM, firmware completo con cámara | 125.8 ms |

Los dos efectos componen (~1.4× por perder la PSRAM, ~2.4× por la cámara):

- **Sin PSRAM los pesos quedan en flash** (no en SRAM: no hay lugar), que es
  el caso lento del experimento 1 — coincide: 59.8 ≈ 60.4 ms.
- **La cámara es el factor dominante**: el DMA del sensor + wifi + HTTP
  compiten por caché, RAM y ciclos con la inferencia (el jitter 114–139 ms
  es la firma). En placas con cámara, presupuestar ~2–2.5× el tiempo del
  benchmark "limpio".
- El modelo chico (18 KB) no pierde nada sin PSRAM (11.5 → 11.6 ms), y
  `standard_rgb96` directamente **no entra** en la S3-CAM sin PSRAM (arena
  de 308 KB contiguos; el esp-dl parcheado loguea el fallo en vez de
  panicquear).

## Backlog

- **Validación end-to-end con cámara**: repetir las pruebas de captura+inferencia
  apuntando a fotos estilo STL-10 (4-5 clases) con el fix del exponente de
  entrada (commit 7882c64). El self-test certifica el camino modelo/cuantización
  (paridad bit a bit); falta validar el camino cámara→preprocess→inferencia en
  condiciones reales. Primera señal positiva: bird 94.4% con entropía 0.12.
- **esp-dl: caída limpia cuando el modelo no entra en RAM**: si el arena de
  activaciones no se puede alocar (ej. good_model 96×96 pide 308 KB contiguos
  y el bloque máximo de RAM interna del S3 es 160 KB), esp-dl deja el modelo
  a medio cargar, `inference_init` reporta éxito y la primera inferencia
  crashea con LoadProhibited. Detectar el fallo y caer a dummy mode con un
  mensaje claro.
- ~~**ESP32-S3: TFLite Micro vs esp-ppq/ESP-DL**: portar uno de los modelos a
  TensorFlow Lite Micro (con los kernels optimizados de esp-nn) y comparar en
  la misma placa contra el camino esp-ppq→ESP-DL actual.~~ **Hecho** — ver
  § "Runtimes alternativos (`firmware_rt/`)": los 4 modelos medidos bajo
  TFLM (con y sin esp-nn), onnx2c y ExecuTorch, con self-test PASS en todos.
- Adaptar las variantes de cuantización b (torchao) y c (brevitas) al esquema
  prune-sin-finetune (deben aceptar como `--source` un checkpoint que ya pasó
  por `script03-finetune.py`, igual que 04a) y agregarlas a `pipelines.yaml`
  como variantes alternativas de los flows existentes (hoy sólo usan
  `variant: a`).

---

## Confianza y entropía

Todos los scripts y el firmware reportan dos métricas sobre el vector softmax:

- **`confidence`** — `max(softmax)` ∈ [0, 1].
- **`entropy_normalized`** — entropía de Shannon normalizada: `H = -Σ pᵢ·log(pᵢ) / log(N)` ∈ [0, 1].
  - `H ≈ 0`: el modelo está seguro (una clase domina).
  - `H ≈ 1`: distribución plana, clasificación no confiable.

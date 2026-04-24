# ESP32-CAM — Flujo completo de compresión y despliegue

Pipeline educativo del curso **CEIA IA Embebida**: diseño de una CNN → entrenamiento → pruning → cuantización → despliegue en un AI-Thinker ESP32-CAM.

---

## Requisitos

- Python 3.10–3.11
- [UV](https://docs.astral.sh/uv/) como gestor de entorno
- Hardware: AI-Thinker ESP32-CAM + ESP32-CAM-MB (solo para el firmware)

```bash
cd examples/esp32-cam
uv sync           # crea .venv e instala dependencias
```

El dataset STL-10 (~2.6 GB) se descarga automáticamente en `examples/datasets/` la primera vez que se ejecuta cualquier script de entrenamiento.

---

## Verificación del hardware

Antes de correr el pipeline completo, conviene confirmar que el módulo ESP32-CAM y su programador MB están funcionando. El subproyecto [`../esp32-cam-test/`](../esp32-cam-test/) contiene un firmware ESP-IDF mínimo para eso.

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

> `script04-build.py` usa `idf.py flash` que también falla sin el cold boot. Si se usa el script, hacer el cold boot antes de lanzarlo o usar `--no-preflight` + cold boot manual.

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
esp32-cam/
├── esp32cam_utils.py               # Utilidades compartidas (modelos, training, pruning)
├── quantize_utils.py               # Infra común a los tres scripts de cuantización
├── script01-train.py               # Entrenamiento (todas las variantes CNN)
├── script02-prune.py               # Pruning estructurado (weight / activation)
├── script03a-quantize.py           # Cuantización PTQ (esp-ppq, sin QAT real)
├── script03b-quantize-torchao.py   # QAT real con torchao + PTQ final con esp-ppq
├── script03c-quantize-brevitas.py  # QAT real con brevitas, puentes preconditioned/qdq
├── models/                         # Checkpoints .pth y exportaciones .espdl
├── outputs/                        # Figuras y JSONs de resultados
└── tests/
    ├── test_utils.py               # Tests de arquitecturas, entropía y pruning
    └── test_quantize.py            # Tests de quantize_utils + smoke tests QAT
```

El paso 3 del flujo tiene tres variantes (a/b/c) que producen todas un `.espdl`
consumible por el firmware — eligés cuál según el método de cuantización que
quieras demostrar. Todas comparten el flujo 01 → 02 → ... anterior.

---

## Flujo de trabajo

```
script01-train  →  script02-prune  →  script03{a|b|c}-quantize*  →  firmware ESP-IDF
(entrena)          (prune)            (cuantiza, tres variantes)   (despliega)
```

Cada script guarda el checkpoint con un nombre que concatena los pasos aplicados:

```
standard_rgb96.pth
depthwise_gray32.pth
depthwise_gray32_pruned_activation_30.pth
depthwise_gray32_pruned_activation_30_quantized_ptq.espdl            # 03a
depthwise_gray32_pruned_activation_30_quantized_qat_torchao.espdl    # 03b
depthwise_gray32_pruned_activation_30_quantized_qat_brevitas.espdl   # 03c preconditioned
depthwise_gray32_pruned_activation_30_quantized_qat_brevitas_qdq.espdl  # 03c qdq
```

Si el checkpoint de salida ya existe, el script lo carga y omite el procesamiento.

---

## Script de entrenamiento (01-train)

Dataset: **STL-10** — 10 clases, 5000 train / 8000 test, 96×96 RGB.

```bash
# Las 5 variantes del flujo educativo:
uv run python script01-train.py --arch standard  --input-mode rgb  --resolution 96  # baseline
uv run python script01-train.py --arch standard  --input-mode rgb  --resolution 32  # menor resolución
uv run python script01-train.py --arch standard  --input-mode gray --resolution 32  # escala de grises
uv run python script01-train.py --arch depthwise --input-mode rgb  --resolution 32  # depthwise
uv run python script01-train.py --arch depthwise --input-mode gray --resolution 32  # candidato a pruning
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

```bash
# Pruning por activación, 30% de canales eliminados, 10 epochs de fine-tuning
uv run python script02-prune.py --source depthwise_gray32 --method activation --ratio 0.3

# Pruning por magnitud de pesos, 50%
uv run python script02-prune.py --source standard_rgb32 --method weight --ratio 0.5

# Sin pruning (baseline para comparar)
uv run python script02-prune.py --source depthwise_gray32 --method none
```

| Flag | Valores | Default |
|------|---------|---------|
| `--source` | nombre del checkpoint (sin `.pth`) | requerido |
| `--method` | `none` \| `weight` \| `activation` | `activation` |
| `--ratio` | fracción de canales a eliminar | `0.3` |
| `--finetune-epochs` | epochs de fine-tuning post-pruning | `10` |
| `--lr` | learning rate del fine-tuning | `2e-4` |
| `--batch-size` | tamaño de lote | `64` |

El checkpoint resultante guarda la topología reducida (número real de canales supervivientes) para que `script03-quantize` pueda reconstruir el modelo correctamente.

---

## Scripts de cuantización (03a / 03b / 03c)

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

### 03a — esp-ppq directo (baseline, sin QAT real)

```bash
# PTQ puro
uv run python script03a-quantize.py --source depthwise_gray32_pruned_activation_30 --mode ptq

# "QAT" = fine-tune sobre el modelo ya PTQ-cuantizado (no es QAT con fake-quant
# real durante training; para eso ver 03b/03c)
uv run python script03a-quantize.py --source depthwise_gray32 --mode qat
```

| Flag | Valores | Default |
|------|---------|---------|
| `--source` | nombre del checkpoint (sin `.pth`) | requerido |
| `--mode` | `ptq` \| `qat` | `ptq` |
| `--calib-batches` | batches de calibración PTQ | `32` |
| `--qat-epochs` | epochs de fine-tune post-PTQ | `5` |
| `--firmware-dir` | destino de `model_data.h` | `firmware/main/` |

### 03b — torchao QAT + esp-ppq PTQ

Inserta fake-quant **real** durante training usando el flujo eager-mode de
`torch.ao.quantization`, entrena contra esos nodos, strip-ea los fake-quant
y pasa el `.pth` resultante por el mismo `esp-ppq` que usa 03a. Los scales
aprendidos se descartan (los re-calibra esp-ppq); lo que sobrevive es la
robustez al ruido de cuantización.

```bash
uv run python script03b-quantize-torchao.py --source depthwise_gray32_pruned_activation_30
uv run python script03b-quantize-torchao.py --source standard_rgb32 --qat-epochs 10
```

| Flag | Valores | Default |
|------|---------|---------|
| `--source` | nombre del checkpoint (sin `.pth`) | requerido |
| `--qat-epochs` | epochs de QAT con fake-quant | `5` |
| `--qat-lr` | learning rate del QAT | `1e-4` |
| `--calib-batches` | batches de calibración PTQ post-strip | `32` |
| `--firmware-dir` | destino de `model_data.h` | `firmware/main/` |

### 03c — brevitas QAT con dos puentes

Usa brevitas (QAT con fake-quant basados en `QuantConv2d`/`QuantLinear`).
Ofrece dos puentes hacia `.espdl`, seleccionables con `--bridge`:

- **`preconditioned`** (default): igual patrón que 03b — QAT con scales float,
  strip, esp-ppq PTQ. Los scales aprendidos se descartan.
- **`qdq`**: QAT con scales **potencia de 2** (coincide con el esquema de
  ESP-DL por construcción), exporta ONNX con nodos QDQ cargando las escalas
  aprendidas, y los pasa a `esppq.espdl_quantize_onnx(target="c")`. Las
  escalas aprendidas en QAT sobreviven end-to-end hasta el `.espdl`.

```bash
# Puente A (preconditioned) — default
uv run python script03c-quantize-brevitas.py --source depthwise_gray32_pruned_activation_30

# Puente B (qdq) — end-to-end con PoT scales
uv run python script03c-quantize-brevitas.py --source depthwise_gray32_pruned_activation_30 --bridge qdq
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
| Ver el baseline sin QAT, comparar contra los otros | **03a** |
| Mostrar QAT con la API nativa de PyTorch (torchao) | **03b** |
| Mostrar QAT con scales PoT (el esquema real de ESP-DL) | **03c --bridge qdq** |
| Un QAT portable que funciona incluso si el qdq bridge falla | **03c --bridge preconditioned** |

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

## Métricas orientativas (gray 32×32)

| Modelo | Params | MACs | KB |
|--------|--------|------|----|
| StandardCNN | ~230 K | ~37 M | ~900 |
| DepthwiseCNN | ~30 K | ~5 M | ~120 |
| Depthwise pruned 30% | ~15 K | ~2.5 M | ~60 |
| Depthwise quantized int8 | ~15 K | ~2.5 M | ~15 |

---

## Entry point: `main.py`

`main.py` orquesta el pipeline completo (train → prune → quantize → build) a partir de flows declarados en `pipelines.yaml`. Cada flow es una lista ordenada de stages self-contained (todos los parámetros declarados explícitamente).

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

| Flow | Stages | Uso |
|------|--------|-----|
| `default` | train → prune → quantize → build | pipeline completo desde cero |
| `quick_redeploy` | quantize → build | re-cuantizar y flashear sin re-entrenar |
| `no_pruning` | train → quantize → build | saltea el stage de pruning |

Definir un nuevo flow es editar `pipelines.yaml` — no requiere tocar Python.

### `pipelines.yaml`

Cada entrada del flow es un diccionario con:

- `stage`: `train` | `prune` | `quantize` | `build`
- `source`: tag del modelo de entrada (requerido para `prune` y `quantize`)
- `variant`: `a` | `b` | `c` (sólo para `quantize`; default `a`)
- `params`: dict de flags que se traducen a `--kebab-case`

Ver el YAML incluido en este directorio para los tres flows de referencia.

### `script04-build.py`

Implementa el stage `build`. Se puede correr standalone:

```bash
cd examples/esp32-cam
uv run python script04-build.py --port /dev/ttyUSB0 --monitor
uv run python script04-build.py --build-only   # sólo compila, no flashea
uv run python script04-build.py --no-preflight # salta los chequeos previos
```

Preflight chequea: `IDF_PATH`, `idf.py` en PATH, `firmware/main/model_data.h` presente, coherencia entre `model_data.h` y `model_data.meta.json`, PSRAM habilitada, y puerto USB accesible.

---

## Firmware ESP-IDF

El firmware del ESP32-CAM vive en [`firmware/`](./firmware/) y consume `model_data.h` generado por `script03-quantize.py`. Soporta:

- Trigger por serie (cualquier byte RX en UART0)
- Trigger y publicación por MQTT (opcionales vía `menuconfig`)
- Salida dual: texto humano + JSON en serie, JSON en `esp32cam/result`

Build rápido:

```bash
cd examples/esp32-cam/firmware
idf.py set-target esp32
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

Ver [`firmware/README.md`](./firmware/README.md) para el procedimiento completo, opciones de menuconfig y tests de host del módulo de preprocesamiento.

---

## Confianza y entropía

Todos los scripts y el firmware reportan dos métricas sobre el vector softmax:

- **`confidence`** — `max(softmax)` ∈ [0, 1].
- **`entropy_normalized`** — entropía de Shannon normalizada: `H = -Σ pᵢ·log(pᵢ) / log(N)` ∈ [0, 1].
  - `H ≈ 0`: el modelo está seguro (una clase domina).
  - `H ≈ 1`: distribución plana, clasificación no confiable.

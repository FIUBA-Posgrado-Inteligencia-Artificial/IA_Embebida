# CLAUDE.md — `examples/esp32-pipelines/`

Guía específica para trabajar dentro de este directorio. Complementa (no reemplaza) al `CLAUDE.md` raíz.

## Layout Python

Los scripts de pipeline y generadores viven en `scripts/`; las utilidades
importables (`esp32cam_utils`, `quantize_utils`, `model_registry`,
`pipeline_runner`) en `pipelib/`. `main.py` queda en la raíz. Los scripts y
`main.py` insertan `pipelib/` en `sys.path`, así que los imports siguen siendo
planos (`from esp32cam_utils import ...`). Todo se ejecuta desde la raíz del
ejemplo: `uv run python scripts/script01-train.py ...`.

## Qué vive acá

Pipeline end-to-end para entrenar, comprimir y desplegar una CNN chica para clasificación de imágenes (STL-10) sobre **ESP32-CAM** vía ESP-DL.

Flujo por stage: **train → prune → finetune → quantize → build → flash → monitor**. `prune` ya no finetunea internamente (ver `script03-finetune.py`); no hay validación de dependencias entre stages — es responsabilidad del usuario mantener el `source:` en sync.

## Entry point: `main.py` + `pipelines.yaml`

Todo se orquesta desde `main.py`. Los flows están **declarados en YAML**, no hardcodeados en Python:

```bash
uv run python main.py run <flow>              # ejecuta un flow
uv run python main.py run <flow> --dry-run    # sólo imprime los comandos
uv run python main.py flows                   # lista flows
uv run python main.py models                  # tabla de modelos en models/
uv run python main.py models clean [--dry-run]# reconcilia models.json con disco
```

Agregar un flow nuevo = editar `pipelines.yaml`. No tocar Python.

Un flow puede heredar de otro con `extends: <flow>` + `stages: [...]` — la
lista efectiva es la del padre seguida de los stages propios. Como los
scripts cachean sus outputs, re-ejecutar los stages heredados es casi gratis.
Los flows se organizan en capas: `<modelo>_base` (train→prune→finetune,
agnóstico del chip) → `<modelo>` (+quantize ESP32) → `<modelo>_esp32`
(+build/flash/monitor) o `<modelo>_esp32s3` (quantize/build del S3 desde
`_base`). El chip va como sufijo (`_esp32`, `_esp32s3`), nunca como prefijo.
Los flows genéricos `flash` y `monitor` no dependen de ningún modelo: operan
sobre lo último buildeado / lo que ya corre en la placa.

## Módulos clave

| Archivo | Responsabilidad | Notas |
|---|---|---|
| `main.py` | CLI (argparse con subparsers) | Despacha a `_cmd_run`/`_cmd_flows`/`_cmd_models`. Refresca registry post-run live (no dry-run). |
| `pipelib/pipeline_runner.py` | Carga/valida YAML, resuelve `extends`, `stage → argv`, `Popen` con fail-fast y prefijo `[stage]` | `VALID_STAGES = {train, prune, finetune, quantize, build, flash, monitor}`. `quantize.variant ∈ {a,b,c}`. `prune`/`finetune`/`quantize` requieren `source`. `extends` valida padre existente y detecta ciclos. El `port` top-level de `pipelines.yaml` se inyecta en `params.port` de los stages `flash`/`monitor` que no lo definan (no en `build`). |
| `scripts/script03-finetune.py` | Fine-tuning explícito, stage separado de `prune` | Output `<source>_ft<N>.pth` (`N` = epochs). No hace pruning ni quantize. |
| `pipelib/model_registry.py` | Scan de `models/`, reconcile, render ASCII | Clasifica por nombre: `.espdl` → `espdl`, `_quantized` → `quantized`, `_ft<N>` (sin `_quantized` después) → `finetuned`, `_pruned_` → `pruned`, sino `baseline`. Escritura atómica (tmp + rename). |
| `pipelib/esp32cam_utils.py` | Dataset STL-10, training loop, modelos, métricas, write_model_data_meta | `STL10_CLASSES`, `_NORM_STATS` son la fuente de verdad de las estadísticas de normalización. |
| `pipelib/quantize_utils.py` | esp-ppq PTQ, export `.espdl`, `generate_model_data_h` | `generate_model_data_h` emite `MODEL_INPUT_MEAN_{0,1,2}` y `MODEL_INPUT_STD_{0,1,2}` cuando `input_channels=3`; sólo `_0` para gray. |
| `scripts/script05-build.py` | Wrapper de `idf.py` con preflight | Preflight chequea: `IDF_PATH`, `idf.py` en PATH, `model_data.h`, coherencia header↔sidecar, PSRAM, puerto serie. |

## Los tres `script04*` (quantize variants)

Son **peers**, no pasos secuenciales — la letra hace explícito que ocupan la misma posición en el pipeline pero usan técnicas distintas. Todos terminan en un `.espdl` consumible por el firmware.

| Variante | Técnica | Framework |
|---|---|---|
| `script04a-quantize.py` | Pure PTQ | esp-ppq |
| `script04b-quantize-torchao.py` | QAT precondicionado → export → PTQ finaliza en esp-ppq | torchao |
| `script04c-quantize-brevitas.py` | QAT precondicionado → export → PTQ finaliza en esp-ppq | brevitas |

Los tres emiten el **mismo sidecar** `model_data.meta.json` con las mismas claves (ver §"Contrato de coherencia"). Si cambiás uno, asegurate que los otros dos sigan equivalentes — hay tests que dependen de eso.

### `--target {c,esp32s3}` — el `.espdl` NO es intercambiable entre placas

Los tres `script04*` aceptan `--target` (default `c`), que se propaga al
exportador de esp-ppq:

| `--target` | Placa | Kernels |
|---|---|---|
| `c` | ESP32 clásico (ESP32-CAM) | C portable |
| `esp32s3` | ESP32-S3 (boards `esp32s3`/`esp32s3cam`) | SIMD del S3 |

El `.espdl` resultante **es distinto y no es intercambiable**. Lo peligroso es
que el firmware del target equivocado **no falla**: carga el modelo, corre la
inferencia y devuelve basura, sin error de compilación ni de runtime. La única
forma de detectarlo es que las predicciones no tengan sentido.

Además, `--target` **no cambia el nombre del archivo generado**: para el mismo
`--source`, target `c` y target `esp32s3` producen `<source>.espdl` y
`<source>_model_data.h` con el mismo nombre. Por eso el flow del S3 usa
`--firmware-dir` para dejar su header dentro del propio proyecto S3, y por eso
`install_model_data` en `script05-build.py` busca **primero** en
`firmware/main/` y sólo después en `models/`.

**El chequeo que realmente cierra el agujero** no es ese orden sino el sidecar:
`model_data.meta.json` lleva la llave **`target`**, y `script05-build.py` tiene
su propia flag `--target` (default `c`) que se valida contra ella antes de
instalar nada. Si no coinciden, el build **falla** con "Target mismatch" en vez
de embeber el modelo equivocado. Detalles importantes:

- Se valida **siempre**, no sólo con `--preflight`. El flow del S3 corre con
  `--build-only` y sin preflight, que es justo donde más importa.
- Un stage `build` **sin `source`** (re-buildear lo ya instalado, patrón que
  existe en `pipelines.yaml`) no pasa por `install_model_data`. Ese camino lo
  cubre `check_installed_target`, que valida el `model_data.meta.json` ya
  presente en `firmware/main/` — pero es parte del **preflight**, así que
  `--no-preflight` lo desactiva.
- Un sidecar sin `target` (artefacto viejo) o directamente ausente **también se
  rechaza**: no hay forma de saber para qué chip está compilado, y regenerarlo
  cuesta segundos.
- Por eso todo stage `build` de un flow no-clásico tiene que declarar `target:`
  en `pipelines.yaml`, igual que su stage `quantize`.

Esto existe porque la prioridad de `firmware_main` sola abría su propia
variante del problema: `--firmware-dir` es una flag pública, así que un
`script04* --firmware-dir firmware/main` corrido una sola vez sobre el proyecto
clásico dejaba ahí un `<source>_model_data.h` que le ganaba para siempre a todo
`quantize` posterior — y ese archivo está gitignoreado, o sea invisible en
`git status`.

### `--firmware-dir`

- `script04*-quantize.py --firmware-dir <dir>`: dónde dejar
  `<source>_model_data.h` + `.meta.json`. Default: `models/`.
- `script05-build.py --firmware-dir <dir>`: qué proyecto ESP-IDF buildear /
  flashear. Default: `firmware/`.

Juntas permiten alimentar proyectos firmware fuera de `firmware/` sin duplicar
nada del pipeline.

### El flow `fast_debug_esp32s3cam`

**`fast_debug_esp32s3cam`** extiende `fast_debug_base` con quantize/build
sobre el proyecto firmware unificado (`firmware/`, board `esp32s3cam` —
sensor OV3660, sin PSRAM funcional en la unidad relevada, ver §"Placas y
configuración"). Todo vive en `firmware/`.

El proyecto legacy `examples/esp32-s3-cam/` que originó este port fue
eliminado del repo (2026-08-14); si hiciera falta recuperar algo, está en el
historial de git (commit "chore(examples): eliminar esp32 (didáctico) y
esp32-s3-cam (legacy)").

## Contrato de coherencia: `model_data.h` ↔ `model_data.meta.json`

Cada `script04*` emite dos archivos que viajan juntos al firmware:

- `firmware/main/model_data.h` — header C con `#define MODEL_*` (consumido por la build ESP-IDF)
- `firmware/main/model_data.meta.json` — sidecar JSON con los mismos valores (consumido por el preflight)

`check_metadata_coherence` (en `script05-build.py`) valida que no haya drift. Claves validadas:

```
num_classes, input_w, input_h, input_channels, input_exponent,
mean[0..input_channels-1], std[0..input_channels-1]
```

`input_exponent` es el exponente de cuantización de la ENTRADA calibrado por
esp-ppq (escala 2^exp; p.ej. -6 → factor 64). El preprocess del firmware
cuantiza con `1 << -MODEL_INPUT_EXPONENT` — **nunca** asumir 128 (2^-7):
si el modelo calibró -6, asumir -7 duplica la entrada y satura a ±1σ (bug
encontrado el 2026-08-12 vía dlogits del self-test).

El sidecar además lleva `classes` y **`target`**. `target` no lo valida
`check_metadata_coherence` (no tiene contraparte en el header) sino
`install_model_data` (camino con `--source`, siempre) y
`check_installed_target` (camino sin `--source`, sólo con preflight), ambos
contra el `--target` del build. Es obligatorio:
`write_model_data_meta` tira `KeyError` si falta.

**Convención `mean`/`std`:** el sidecar guarda los valores ya multiplicados por 255 (uint8), para que match literal con los `#define MODEL_INPUT_MEAN_0` del header (que también son uint8). No guardar floats `[0,1]` en el sidecar.

Si agregás un nuevo `MODEL_*` al header, hay que:
1. Agregarlo a `_HEADER_INT_KEYS` en `script05-build.py`
2. Emitirlo también en el sidecar vía `write_model_data_meta` en los tres `script04*`
3. Agregar test en `tests/test_script05_build.py`

## `script05-build.py` preflight

Orden de chequeos (fail-fast, `raise PreflightError`):

1. `check_idf_env` — `IDF_PATH` seteado y `idf.py` en PATH
2. `check_model_data_present` — `firmware/main/model_data.h` existe
3. `check_installed_target` — el sidecar del header **ya instalado** declara el `--target` que se está buildeando (warn si falta sidecar). Cubre el stage `build` **sin** `source`; el camino con `source` lo cubre `install_model_data`, que valida siempre
4. `check_metadata_coherence` — header↔sidecar match (warn si falta sidecar)
5. `check_psram_enabled` — soporte de PSRAM habilitado: `CONFIG_ESP32_SPIRAM_SUPPORT=y` (IDF v4) o `CONFIG_SPIRAM=y` (IDF v5/S3) en `sdkconfig` o `sdkconfig.defaults`. Valida la **config**, no el hardware
5. `resolve_port` — si `--port auto`, scan de `/dev/ttyUSB*` y `/dev/ttyACM*`; error si 0 o >1

Saltable con `--no-preflight`. La flag `--yes` existe pero hoy no tiene efecto (reservada para prompts interactivos futuros).

## Tests

Corren con `uv run pytest tests/ -v` desde `examples/esp32-pipelines/`.

Patrón para testear un script con guiones en el nombre (no importable directo):

```python
import importlib.util
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "script05-build.py"

def _load_module():
    spec = importlib.util.spec_from_file_location("script05_build", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

Patrón para testear el CLI como subprocess: ver `_main()` helper en `tests/test_main_cli.py` — usa path absoluto a `main.py` para que funcione con cualquier cwd.

**Mockear `shutil.which`** para tests de preflight IDF: `monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)`.

## Convenciones de nombres de modelos

Los scripts generan modelos con nombres estructurados **cortos** que
`infer_stage_from_name` sabe clasificar (abreviados el 2026-08-20: las cadenas
de pruning iterativo con nombres largos superaban el límite de 255 bytes por
nombre del filesystem):

- `dw_gray32.pth` → baseline (`std` = standard, `dw` = depthwise)
- `dw_gray32_pra30.pth` → pruned (`_pr` + inicial del método `a`ctivation/`w`eight/`n`one + ratio %)
- `dw_gray32_pra30_ft10.pth` → finetuned (`script03-finetune.py`, `10` = epochs)
- `dw_gray32_pra30_ft10_qptq.pth` → quantized (`_q` + modo; QAT: `_qtao` torchao, `_qbrv` brevitas)
- `dw_gray32_pra30_ft10_qptq.espdl` → espdl

El clasificador y `_quantized_parent_name` siguen aceptando los nombres legacy
(`standard_`, `_pruned_activation_30`, `_quantized_ptq`) por si aparece un
artefacto viejo, pero todo lo nuevo usa la forma corta. `script02-prune.py`
acepta `--output` para renombrar su checkpoint de salida (hoy sin uso en los
flows; quedó como escape para cadenas que excedan el límite del filesystem).

Los `source:` en `pipelines.yaml` son **strings que el usuario debe mantener en sync manualmente** con lo que produce el stage anterior. No hay validación cruzada. Si renombrás el output de un script, actualizá el `source:` del siguiente.

## Dataset

STL-10 se descarga a `examples/datasets/` (cache compartido con el resto del curso). `get_dataloaders()` en `esp32cam_utils.py` lo resuelve vía `Path(__file__).parent.parent / "datasets"`.

## Firmware

Hay **dos proyectos firmware** que se mantienen en paralelo:

| Directorio | Flash path | Partición | Uso |
|---|---|---|---|
| `firmware/` | USB serial (`script05-build.py --board <b>`) | single `factory` (4MB) | Proyecto único **multi-placa** (`esp32`, `esp32cam`, `esp32s3`, `esp32s3cam`), config por header (ver §"Placas y configuración"). Flash convencional — requiere botón BOOT físico en cada flasheo (la ESP32-CAM-MB de AI-Thinker no trae circuito auto-reset). |
| `firmware_test_webserver/` | USB serial, `idf.py flash monitor` | single `factory` 3MB | Isolation test de wifi + cámara + HTTP MJPEG. Sin inferencia ni MQTT; si esto no anda, el stack base está roto. Proyecto aparte, **no** migrado a la config nueva: sigue usando el `wifi_creds.h` gitignoreado (`wifi_creds.example.h` como plantilla). |

`firmware/` ya no usa Kconfig propio ni `wifi_creds.h`: toda su
configuración vive en `firmware/main/firmware_config.h`, trackeado en el
repo (credenciales WiFi incluidas — decisión explícita del usuario). El
`menuconfig` que expone `main.py`/`script05-build.py` sigue existiendo, pero
sólo para config IDF pura (particiones, log level, etc.), no para features
del firmware.

Ambos proyectos son ESP-IDF estándar (v5.3+, PSRAM habilitada donde la placa
la tiene). `firmware/main/model_data.h` lo genera `script04*` — **no editar
a mano**.

## Placas y configuración

`firmware/` es un **único proyecto** parametrizado por placa en build-time
(`idf.py -DBOARD=<board>`, default `esp32cam`) y por features en un header
trackeado. Sin Kconfig propio.

| board | chip IDF (`set-target`) | target `.espdl` | build dir | cámara |
|---|---|---|---|---|
| `esp32` | esp32 | c | `build_esp32/` | no (imagen fija, `static_frame.h`) |
| `esp32cam` | esp32 | c | `build_esp32cam/` | sí (AI-Thinker) |
| `esp32s3` | esp32s3 | esp32s3 | `build_esp32s3/` | no (imagen fija; devkit N16R8: 16MB flash + 8MB PSRAM octal OK, relevada 2026-08-13; el proyecto usa 4MB de flash — hay 12MB sin aprovechar) |
| `esp32s3cam` | esp32s3 | esp32s3 | `build_esp32s3cam/` | sí (OV3660; la placa declara `BOARD_HAS_PSRAM 1`, pero la PSRAM de la unidad relevada está rota — se maneja forzando `FEATURE_PSRAM 0` **manualmente** en `firmware_config.h`, no es el default) |

Esta tabla vive **por duplicado** en `script05-build.py` (`BOARDS`) y en
`firmware/CMakeLists.txt` (`VALID_BOARDS`) — si agregás una placa, actualizá
ambas.

**Flasheo de las placas S3** (relevado 2026-08-13): usar el puerto
**USB-UART** (CH343, `1a86:55d3`) — tiene auto-reset y flashea sin tocar
botones. El puerto USB nativo/OTG (`303a:4001`, CDC de la app corriendo) NO
sirve para esptool: hay que entrar a download mode a mano y se re-enumera.
Ojo con el **cable**: con un cable USB-C marginal el auto-reset de la
ESP32-S3-CAM no funcionaba (esptool no conectaba); con otro cable, anda —
si el auto-reset falla, probar otro cable antes de sospechar de la placa.

- **`firmware/main/board.h`** — punto de entrada único. Incluye
  `boards/board_<nombre>.h` según el `BOARD_<NOMBRE>` que inyecta CMake
  (uno y exactamente uno debe estar definido), y después
  `firmware_config.h` + `firmware_config_check.h`. **Nunca** incluir
  `boards/*.h` ni `firmware_config.h` directo desde código de aplicación —
  siempre `board.h`.
- **`firmware/main/boards/board_<nombre>.h`** — lo intrínseco de la placa,
  no configurable por el usuario: `BOARD_NAME`, `BOARD_HAS_CAMERA`,
  `BOARD_HAS_PSRAM`, pines de cámara si aplica. `esp32`/`esp32s3` tienen
  `BOARD_HAS_CAMERA 0` y sin pines.
- **`firmware/main/firmware_config.h`** — features configurables por el
  usuario, **trackeado en git a propósito** (credenciales WiFi incluidas):
  `FEATURE_WIFI`, `WIFI_SSID`/`WIFI_PASSWORD`, `FEATURE_WEBSERVER` (requiere
  `FEATURE_WIFI`), `FEATURE_HW_VALIDATION`, `FEATURE_PSRAM` (-1 auto vía
  `BOARD_HAS_PSRAM`, 0/1 fuerza), `HTTP_PORT`, `RESIZE_BILINEAR`,
  `START_IN_CONTINUOUS`, `ENTROPY_WARN_THRESHOLD`.
- **`firmware/main/firmware_config_check.h`** — reglas cruzadas placa↔features,
  incluido por `board.h`: `FEATURE_WEBSERVER && !FEATURE_WIFI` corta con
  `#error` en tiempo de compilación (no hay forma de correr el webserver sin
  wifi). Deriva `EFFECTIVE_PSRAM` (el valor real que consume el resto del
  firmware) a partir de `FEATURE_PSRAM`/`BOARD_HAS_PSRAM`.
- **PSRAM override**: `check_psram_enabled` en `script05-build.py` lee el
  `EFFECTIVE_PSRAM` efectivo (placa + `FEATURE_PSRAM`) antes de exigir
  `CONFIG_SPIRAM`/`CONFIG_ESP32_SPIRAM_SUPPORT` en el sdkconfig — si el
  efectivo es 0 (ej. la ESP32-S3-CAM del usuario, con PSRAM rota en HW), el
  preflight no exige nada.
- **`script05-build.py --board {esp32,esp32cam,esp32s3,esp32s3cam}`**
  (default `esp32cam`) deriva chip, build dir y target `.espdl` esperado de
  la tabla `BOARDS`; `--target` ya no existe como flag independiente.
  `flash`/`monitor` también toman `--board` (para el build dir y el
  `--chip` de esptool).
- **Matriz de builds**: `tools/build_matrix.sh` compila las 4 placas de punta
  a punta (`. ~/esp/esp-idf/export.sh && tools/build_matrix.sh`), fail-fast.
  Es el smoke test de referencia tras tocar `board.h`/`firmware_config*.h`/
  `CMakeLists.txt`; no cubre las combinaciones de features (eso se verifica
  a mano editando `firmware_config.h` temporalmente y revirtiendo).

### `firmware/main/capture_state.{c,h}` — estado compartido de captura/inferencia

Módulo central (thread-safe) que guarda el último resultado de inferencia, el
JPEG capturado y el modo activo, para que el trigger serie, el código MQTT
(compilado out por defecto, ver §"Placas y configuración") y el servidor
HTTP lean/escriban un único estado consistente.

- **Modos**: `CAPTURE_MODE_SINGLE` (una captura por trigger) y
  `CAPTURE_MODE_CONTINUOUS` (loop libre, reporta `fps`). Se togglea con la
  tecla `m` por serie o el checkbox "continuous" de la página web
  (`POST /mode` con `continuous=0|1`).
- **Trigger**: en modo single, cualquier byte RX en UART0 que no sea `m`/`t`
  dispara una captura (igual que antes); en modo continuous el firmware
  captura en loop sin esperar trigger.
- **Self-test (tecla `t`)**: corre `inference_run()` sobre golden images del
  test set de STL-10 embebidas en `firmware/main/test_images.h` (gitignoreado)
  y compara contra la referencia del modelo cuantizado en PC. El header lo
  emite `script04a-quantize.py` (`<source>_test_images.h`, con el **mismo
  grafo** que exporta el `.espdl`) y lo instala `script05-build.py` junto con
  `model_data.h` — si el modelo instalado no trae el suyo, el instalador borra
  el stale y el self-test queda deshabilitado (el firmware compila igual).
  Las golden images se eligen con margen top1-top2 ≥ 0.15 en el modelo
  cuantizado, porque el simulador de esp-ppq y los kernels de ESP-DL difieren
  en redondeos: con margen, el PASS esperable es N/N estricto; un FAIL es un
  bug real. `gen_test_images.py` standalone existe para experimentos (usa un
  PTQ propio: referencia no exacta).
- **Servidor HTTP** (`http_stream_server.c`): además del stream MJPEG,
  expone:
  - `GET /` — página con la última captura como vista previa (en continuo se
    refresca sola a ~3 fps), botón de trigger y checkbox de modo continuo.
    **No usa `/stream`** a propósito.
  - `GET /stream` — MJPEG (si la cámara está habilitada). Solo para debug
    manual: su handler es un loop infinito, y como `esp_http_server` atiende
    todo en una sola tarea, mientras hay un cliente conectado bloquea el resto
    de los endpoints **y** el mutex de la cámara (las capturas por serie
    esperan hasta que el cliente se desconecta).
  - `POST /trigger` — dispara una captura+inferencia (equivalente al trigger serie).
  - `POST /mode` — cambia el modo (`continuous=0|1`).
  - `GET /last.jpg` — último JPEG capturado.
  - `GET /result` — JSON con el último resultado; la página lo poll-ea cada 500 ms. Incluye `seq`, `mode`, `last_error`, `top1`, `probs`, `class_names`, `t_capture_ms`, `t_preprocess_ms`, `t_inference_ms`, `fps`.

El port a ESP32-S3 con cámara vive como board `esp32s3cam` del proyecto
unificado (`firmware/`, flow `fast_debug_esp32s3cam`); el proyecto legacy
`examples/esp32-s3-cam/` fue eliminado del repo (ver §"El flow
`fast_debug_esp32s3cam`").

> Nota: existió una variante OTA (`firmware_ota/` + `script04b-build_ota.py` + `script04c-build_test_ota.py`), eliminada por no usarse. Si hiciera falta recuperarla, está en el historial de git (borrada en el commit "chore(esp32-cam): eliminar variante OTA").

## Runtimes alternativos (`firmware_rt/`)

Comparativa de runtimes de inferencia en ESP32-S3 (devkit, sin cámara),
paralela al camino principal esp-ppq→ESP-DL. Resultados y guía de
regeneración en el README (§ "Runtimes alternativos"). Mapa:

- `firmware_rt/common/` — componente compartido `selftest_runner.{c,h}`:
  imprime `ready`, menú serie `t` (self-test) / `c` (una inferencia), y el
  formato de línea que parsea `script06-bench.py`.
- `firmware_rt/tflm/` — TFLite Micro (esp-tflite-micro), int8. **Dos builds**:
  `build_espnn/` (esp-nn SIMD, default) y `build_ref/` (C portable vía
  `CONFIG_NN_ANSI_C=y` en `sdkconfig.ref` — es el Kconfig de esp-nn el que
  apaga el SIMD, no una flag propia). Verificar con `nm` que `build_ref` no
  linkea `esp_nn_*_esp32s3`.
- `firmware_rt/onnx2c/` — C float plano generado por onnx2c. Para rgb96 los
  buffers intermedios van a PSRAM (`EXT_RAM_BSS_ATTR`, lo inyecta su
  `install_model.sh`).
- `firmware_rt/executorch/` — runtime portable de ExecuTorch v1.4.0 (float),
  soporte ESP-IDF upstream (preset `esp-baremetal`). Gotchas: ICE del gcc
  Xtensa con `-O2` en `op_sum.cpp` (usar `EXECUTORCH_OPTIMIZE_SIZE=ON`) y NO
  duplicar el `--whole-archive` de `portable_ops_lib` (ya viene en
  `INTERFACE_LINK_OPTIONS`; duplicado = doble registro de kernels + crash al
  boot).
- `tools/rt_export/{tflite,onnx2c,executorch}/` — exports en PC, cada uno con
  su venv aislado (`uv run --project ...`). Generan header de modelo + golden
  images + logits de referencia con la MISMA toolchain en PC.

Semántica de `tol` en `selftest_runner`:

- `tol = 0` — int8 bit-exacto (espdl: referencia del mismo grafo esp-ppq).
- `tol = -N` — int8 con hasta N LSB de diferencia (TFLM usa `-1`: el requant
  de esp-nn/TFLM redondea distinto que el interpreter LiteRT de x86).
- `tol > 0` — float: PASS si `max|Δlogit| < tol` (onnx2c y executorch usan
  `0.01` contra onnxruntime / runtime ET de PC; dan dlogits=0).

Los benches se registran en `benchmarks.json` bajo el nombre
`<source>_quantized_ptq` (aunque los runtimes float no usan el `.espdl`),
con el runtime como parte de la key: `script06-bench.py --runtime
{espdl,tflm,tflm_espnn,onnx2c,executorch} --build-dir <dir> --skip-build`.

## Qué no hacer

- No incluir `boards/*.h` ni `firmware_config.h` directo desde código de
  aplicación — siempre `board.h` (es el único que valida que haya una y sólo
  una placa activa y arma el orden correcto de includes).
- No agregar variantes `script04*` sin replicar el sidecar `model_data.meta.json` — romperías el preflight.
- No poner floats `[0,1]` en el sidecar; van como uint8 (`int(x*255)`).
- No editar `model_data.h` a mano: se regenera cada quantize.
- No agregar un stage `build` para una placa no clásica sin su `board:` en
  `pipelines.yaml` — sin eso el build valida contra el default `esp32cam` y
  rechaza (o peor, acepta) el modelo equivocado.
- **No asumir que un `.espdl` sirve para las dos placas.** El de `--target c`
  (ESP32 clásico) y el de `--target esp32s3` tienen el mismo nombre pero son
  binarios distintos; cruzarlos no da error, da predicciones basura en
  silencio. Regenerar siempre con el flow que corresponde a la placa.
- No hardcodear flows en Python; van en `pipelines.yaml`.
- No mezclar mock/real del database — los tests unitarios de módulos puros usan objetos reales sobre `tmp_path`; sólo los tests de integración de `main.py` usan subprocess.

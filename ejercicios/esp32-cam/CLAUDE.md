# CLAUDE.md — `examples/esp32-cam/`

Guía específica para trabajar dentro de este directorio. Complementa (no reemplaza) al `CLAUDE.md` raíz.

## Qué vive acá

Pipeline end-to-end para entrenar, comprimir y desplegar una CNN chica para clasificación de imágenes (STL-10) sobre **ESP32-CAM** vía ESP-DL.

Flujo por stage: **train → prune → quantize → build (firmware ESP-IDF)**.

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

## Módulos clave

| Archivo | Responsabilidad | Notas |
|---|---|---|
| `main.py` | CLI (argparse con subparsers) | Despacha a `_cmd_run`/`_cmd_flows`/`_cmd_models`. Refresca registry post-run live (no dry-run). |
| `pipeline_runner.py` | Carga/valida YAML, `stage → argv`, `Popen` con fail-fast y prefijo `[stage]` | `VALID_STAGES = {train, prune, quantize, build}`. `quantize.variant ∈ {a,b,c}`. `prune`/`quantize` requieren `source`. |
| `model_registry.py` | Scan de `models/`, reconcile, render ASCII | Clasifica por nombre: `.espdl` → `espdl`, `_quantized` → `quantized`, `_pruned_` → `pruned`, sino `baseline`. Escritura atómica (tmp + rename). |
| `esp32cam_utils.py` | Dataset STL-10, training loop, modelos, métricas, write_model_data_meta | `STL10_CLASSES`, `_NORM_STATS` son la fuente de verdad de las estadísticas de normalización. |
| `quantize_utils.py` | esp-ppq PTQ, export `.espdl`, `generate_model_data_h` | `generate_model_data_h` emite `MODEL_INPUT_MEAN_{0,1,2}` y `MODEL_INPUT_STD_{0,1,2}` cuando `input_channels=3`; sólo `_0` para gray. |
| `script04-build.py` | Wrapper de `idf.py` con preflight | Preflight chequea: `IDF_PATH`, `idf.py` en PATH, `model_data.h`, coherencia header↔sidecar, PSRAM, puerto serie. |
| `script04b-build_ota.py` | Variante OTA de script04 — apunta a `firmware_ota/` y sube el `.bin` por HTTP POST a `http://<ip>/ota`. | Comparte helpers con `script04-build.py` vía `importlib.util`. Modos: default (build+upload, requiere `--ip`), `--initial` (primer USB flash), `--build-only`, `--info-only` (GET `/ota/info`), `--menuconfig-only`. |

## Los tres `script03*` (quantize variants)

Son **peers**, no pasos secuenciales — la letra hace explícito que ocupan la misma posición en el pipeline pero usan técnicas distintas. Todos terminan en un `.espdl` consumible por el firmware.

| Variante | Técnica | Framework |
|---|---|---|
| `script03a-quantize.py` | Pure PTQ | esp-ppq |
| `script03b-quantize-torchao.py` | QAT precondicionado → export → PTQ finaliza en esp-ppq | torchao |
| `script03c-quantize-brevitas.py` | QAT precondicionado → export → PTQ finaliza en esp-ppq | brevitas |

Los tres emiten el **mismo sidecar** `model_data.meta.json` con las mismas claves (ver §"Contrato de coherencia"). Si cambiás uno, asegurate que los otros dos sigan equivalentes — hay tests que dependen de eso.

## Contrato de coherencia: `model_data.h` ↔ `model_data.meta.json`

Cada `script03*` emite dos archivos que viajan juntos al firmware:

- `firmware/main/model_data.h` — header C con `#define MODEL_*` (consumido por la build ESP-IDF)
- `firmware/main/model_data.meta.json` — sidecar JSON con los mismos valores (consumido por el preflight)

`check_metadata_coherence` (en `script04-build.py`) valida que no haya drift. Claves validadas:

```
num_classes, input_w, input_h, input_channels,
mean[0..input_channels-1], std[0..input_channels-1]
```

**Convención `mean`/`std`:** el sidecar guarda los valores ya multiplicados por 255 (uint8), para que match literal con los `#define MODEL_INPUT_MEAN_0` del header (que también son uint8). No guardar floats `[0,1]` en el sidecar.

Si agregás un nuevo `MODEL_*` al header, hay que:
1. Agregarlo a `_HEADER_INT_KEYS` en `script04-build.py`
2. Emitirlo también en el sidecar vía `write_model_data_meta` en los tres `script03*`
3. Agregar test en `tests/test_script04_build.py`

## `script04-build.py` preflight

Orden de chequeos (fail-fast, `raise PreflightError`):

1. `check_idf_env` — `IDF_PATH` seteado y `idf.py` en PATH
2. `check_model_data_present` — `firmware/main/model_data.h` existe
3. `check_metadata_coherence` — header↔sidecar match (warn si falta sidecar)
4. `check_psram_enabled` — `CONFIG_ESP32_SPIRAM_SUPPORT=y` en `sdkconfig` o `sdkconfig.defaults`
5. `resolve_port` — si `--port auto`, scan de `/dev/ttyUSB*` y `/dev/ttyACM*`; error si 0 o >1

Saltable con `--no-preflight`. La flag `--yes` existe pero hoy no tiene efecto (reservada para prompts interactivos futuros).

## Tests

Corren con `uv run pytest tests/ -v` desde `examples/esp32-cam/`.

Patrón para testear un script con guiones en el nombre (no importable directo):

```python
import importlib.util
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "script04-build.py"

def _load_module():
    spec = importlib.util.spec_from_file_location("script04_build", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

Patrón para testear el CLI como subprocess: ver `_main()` helper en `tests/test_main_cli.py` — usa path absoluto a `main.py` para que funcione con cualquier cwd.

**Mockear `shutil.which`** para tests de preflight IDF: `monkeypatch.setattr("shutil.which", lambda name: "/tmp/idf.py" if name == "idf.py" else None)`.

## Convenciones de nombres de modelos

Los scripts generan modelos con nombres estructurados que `infer_stage_from_name` sabe clasificar:

- `depthwise_gray32.pth` → baseline
- `depthwise_gray32_pruned_activation_30.pth` → pruned
- `depthwise_gray32_pruned_activation_30_quantized.pth` → quantized
- `depthwise_gray32_pruned_activation_30_quantized.espdl` → espdl

Los `source:` en `pipelines.yaml` son **strings que el usuario debe mantener en sync manualmente** con lo que produce el stage anterior. No hay validación cruzada. Si renombrás el output de un script, actualizá el `source:` del siguiente.

## Dataset

STL-10 se descarga a `examples/datasets/` (cache compartido con el resto del curso). `get_dataloaders()` en `esp32cam_utils.py` lo resuelve vía `Path(__file__).parent.parent / "datasets"`.

## Firmware

Hay **dos proyectos firmware** que se mantienen en paralelo:

| Directorio | Flash path | Partición | Uso |
|---|---|---|---|
| `firmware/` | USB serial (`script04-build.py`) | single `factory` (4MB) | Flash convencional — requiere botón BOOT físico en cada flasheo (la ESP32-CAM-MB de AI-Thinker no trae circuito auto-reset). |
| `firmware_ota/` | USB la 1ra vez, HTTP POST `/ota` después (`script04b-build_ota.py`) | dual `ota_0`/`ota_1` (2×1.94MB) + `otadata` | Desarrollo iterativo sin tocar botones — tras el primer `--initial` USB-flash, cada build nuevo se sube inalámbrico. |

Ambos proyectos son ESP-IDF estándar (v5.3+, PSRAM habilitada, board ESP32-CAM con chip ESP32 clásico). `main/model_data.h` lo genera `script03*` en ambos — **no editar a mano**.

### Archivos que divergen entre los dos firmwares

| Archivo | firmware/ | firmware_ota/ |
|---|---|---|
| `partitions.csv` | `factory` 4MB | `otadata` + `ota_0` + `ota_1` |
| `main/Kconfig.projbuild` | sin `ESP32CAM_OTA_ENABLED` | agrega `ESP32CAM_OTA_ENABLED` (default y, depende de `ESP32CAM_HTTP_STREAM_ENABLED`) |
| `main/http_stream_server.{c,h}` | handle local a la función | handle module-static + getter `http_stream_server_get_handle()` expuesto en el `.h` |
| `main/ota_handler.{c,h}` | no existe | implementa `POST /ota` (recibe binario crudo, escribe vía `esp_ota_ops`, reinicia) y `GET /ota/info` (running/boot/update_target) |
| `main/CMakeLists.txt` | sin `app_update` | agrega `ota_handler.c` y `app_update` + `esp_partition` a REQUIRES |
| `main/app_main.c` | sólo `http_stream_server_start()` | también `ota_handler_register(http_stream_server_get_handle())` |

Los archivos de modelo y lógica de inferencia (`inference.cpp`, `camera.c`, `preprocess.c`, etc.) son idénticos en ambos directorios — si los modificás en uno, copialos al otro.

### Endpoint OTA (firmware_ota/)

```
POST /ota               cuerpo = binario crudo del app .bin (Content-Type irrelevante)
                        → escribe en ota_next, esp_ota_set_boot_partition, esp_restart()
                        → responde JSON { ok, bytes, next_partition, rebooting_in_ms }

GET  /ota/info          → JSON { running, boot, update_target, update_size }
```

Sin auth — asumido uso en red local de aula. Para producción agregar basic auth o shared-secret header.

## Qué no hacer

- No agregar variantes `script03*` sin replicar el sidecar `model_data.meta.json` — romperías el preflight.
- No poner floats `[0,1]` en el sidecar; van como uint8 (`int(x*255)`).
- No editar `model_data.h` a mano: se regenera cada quantize.
- No hardcodear flows en Python; van en `pipelines.yaml`.
- No mezclar mock/real del database — los tests unitarios de módulos puros usan objetos reales sobre `tmp_path`; sólo los tests de integración de `main.py` usan subprocess.
- No modificar `firmware/` y `firmware_ota/` divergiendo en archivos que no están listados en la tabla "Archivos que divergen" — cambios a `inference.cpp`, `camera.c`, etc. deben replicarse en ambos.
- No cambiar el `partitions.csv` de `firmware_ota/` a single-factory: perderías la capacidad OTA y la primera subida dejaría al device en un estado inconsistente (otadata apuntando a una partición que ya no existe).

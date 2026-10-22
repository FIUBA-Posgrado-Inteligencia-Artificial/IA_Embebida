# Firmware Multi-Placa + Config por Header — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un único `firmware/` parametrizado para 4 placas (esp32, esp32cam, esp32s3, esp32s3cam) con features en `firmware_config.h` trackeado, seleccionando placa vía `idf.py -DBOARD=...` desde el pipeline.

**Architecture:** Headers de placa (`boards/board_<name>.h`) con lo intrínseco (cámara, pines, PSRAM default) + `firmware_config.h` con features del usuario (wifi, webserver, hw-validation, psram override, parámetros) + `firmware_config_check.h` con reglas `#error`. `script05-build.py` deriva chip IDF, target `.espdl` y build dir de una tabla board→config. El Kconfig propio se elimina.

**Tech Stack:** ESP-IDF 5.3 (C, CMake), Python 3 + pytest.

**Spec:** `docs/superpowers/specs/2026-08-13-firmware-multiboard-config-design.md`

## Global Constraints

- Cwd: `examples/esp32-cam/`. Tests: `uv run pytest tests/ -v`. Build de referencia: `bash -c '. ~/esp/esp-idf/export.sh >/dev/null 2>&1 && cd firmware && idf.py -B build_esp32cam -DBOARD=esp32cam build'`.
- Commits SIEMPRE por paths explícitos (`git commit -m ... -- <paths>`); NUNCA tocar/commitear `examples/esp32-s3-cam/` (solo LEER su camera.c para copiar pines), `examples/esp32/`, `firmware_test_webserver/`, ni el CLAUDE.md raíz.
- Boards válidos y su derivación (tabla única, usar EXACTAMENTE estos valores):
  `esp32 → chip esp32, espdl c` · `esp32cam → chip esp32, espdl c` · `esp32s3 → chip esp32s3, espdl esp32s3` · `esp32s3cam → chip esp32s3, espdl esp32s3`; build dir `build_<board>/`.
- Reglas de config: webserver requiere wifi (`#error`); `BOARD_HAS_CAMERA` manda (no hay flag de usuario); `FEATURE_PSRAM -1|0|1` (-1 = `BOARD_HAS_PSRAM`).
- Credenciales WiFi en `firmware_config.h`, TRACKEADO (decisión explícita del usuario): SSID `TeleCentro-fecf`, password `sinlugaradudas`.
- MQTT fuera de la config nueva: `esp32cam_mqtt.c` queda, gated por `CONFIG_ESP32CAM_MQTT_ENABLED` que ya no define nadie (compilado out). No borrarlo.
- El binario para `esp32cam` debe seguir pasando el self-test 10/10 (no romper el contrato model_data/test_images).

---

### Task 1: Board headers + selección por CMake + firmware_config.h

**Files:**
- Create: `firmware/main/boards/board_esp32.h`, `firmware/main/boards/board_esp32cam.h`, `firmware/main/boards/board_esp32s3.h`, `firmware/main/boards/board_esp32s3cam.h`, `firmware/main/board.h`, `firmware/main/firmware_config.h`, `firmware/main/firmware_config_check.h`
- Modify: `firmware/CMakeLists.txt`, `firmware/main/CMakeLists.txt`
- Test: build (este task solo agrega headers; nada los incluye aún)

**Interfaces:**
- Produces: macros `BOARD_NAME` (string), `BOARD_HAS_CAMERA` (0/1), `BOARD_HAS_PSRAM` (0/1), `BOARD_CAM_PIN_*` y `BOARD_CAM_XCLK_FREQ_HZ` (solo si cámara); `FEATURE_WIFI`, `WIFI_SSID`, `WIFI_PASSWORD`, `FEATURE_WEBSERVER`, `FEATURE_HW_VALIDATION`, `FEATURE_PSRAM`, `EFFECTIVE_PSRAM`, `HTTP_PORT`, `RESIZE_BILINEAR`, `START_IN_CONTINUOUS`, `ENTROPY_WARN_THRESHOLD`. Todo vía `#include "board.h"` (incluye board + config + check).

- [ ] **Step 1: Headers de placa**

`boards/board_esp32cam.h` (pines = los actuales de `firmware/main/camera.c:29-44`, xclk 20 MHz):

```c
#pragma once
#define BOARD_NAME        "esp32cam"
#define BOARD_HAS_CAMERA  1
#define BOARD_HAS_PSRAM   1
#define BOARD_CAM_XCLK_FREQ_HZ 20000000
#define BOARD_CAM_PIN_PWDN  32
#define BOARD_CAM_PIN_RESET -1
#define BOARD_CAM_PIN_XCLK   0
#define BOARD_CAM_PIN_SIOD  26
#define BOARD_CAM_PIN_SIOC  27
#define BOARD_CAM_PIN_D7    35
#define BOARD_CAM_PIN_D6    34
#define BOARD_CAM_PIN_D5    39
#define BOARD_CAM_PIN_D4    36
#define BOARD_CAM_PIN_D3    21
#define BOARD_CAM_PIN_D2    19
#define BOARD_CAM_PIN_D1    18
#define BOARD_CAM_PIN_D0     5
#define BOARD_CAM_PIN_VSYNC 25
#define BOARD_CAM_PIN_HREF  23
#define BOARD_CAM_PIN_PCLK  22
```

`boards/board_esp32s3cam.h`: LEER (no modificar) `../esp32-s3-cam/firmware/main/camera.c` y copiar sus `CAM_PIN_*` y su `xclk_freq_hz` con el mismo formato (`BOARD_CAM_PIN_PWDN -1`, XCLK 15, SIOD 4, SIOC 5, D7..D0 = 16,17,18,12,10,8,9,11, VSYNC 6, HREF 7, PCLK 13 — verificar xclk en el archivo). `BOARD_HAS_PSRAM 1` con comentario: "la placa relevada tiene la PSRAM rota — forzar FEATURE_PSRAM 0 en firmware_config.h".

`boards/board_esp32.h` y `boards/board_esp32s3.h`:

```c
#pragma once
#define BOARD_NAME        "esp32"      /* o "esp32s3" */
#define BOARD_HAS_CAMERA  0
#define BOARD_HAS_PSRAM   1            /* esp32s3: 0 (los devkit S3 del curso no traen) */
```

(Para `board_esp32.h` usar `BOARD_HAS_PSRAM 1` — el chip relevado tiene 8MB; para `board_esp32s3.h` usar 0.)

- [ ] **Step 2: `board.h` (selector) y `firmware_config_check.h`**

```c
/* board.h — punto de entrada único de configuración. Incluir SIEMPRE este,
 * nunca boards/*.h ni firmware_config.h directo. */
#pragma once
#if defined(BOARD_ESP32)
#include "boards/board_esp32.h"
#elif defined(BOARD_ESP32CAM)
#include "boards/board_esp32cam.h"
#elif defined(BOARD_ESP32S3)
#include "boards/board_esp32s3.h"
#elif defined(BOARD_ESP32S3CAM)
#include "boards/board_esp32s3cam.h"
#else
#error "BOARD no definido: buildear con idf.py -DBOARD=<esp32|esp32cam|esp32s3|esp32s3cam>"
#endif
#include "firmware_config.h"
#include "firmware_config_check.h"
```

```c
/* firmware_config_check.h — reglas entre placa y features. */
#pragma once
#if FEATURE_WEBSERVER && !FEATURE_WIFI
#error "FEATURE_WEBSERVER requiere FEATURE_WIFI (firmware_config.h)"
#endif
#if FEATURE_PSRAM == -1
#define EFFECTIVE_PSRAM BOARD_HAS_PSRAM
#else
#define EFFECTIVE_PSRAM FEATURE_PSRAM
#endif
```

- [ ] **Step 3: `firmware_config.h`** — contenido EXACTO de la sección 2 del spec (wifi TeleCentro-fecf / sinlugaradudas, `FEATURE_WIFI 1`, `FEATURE_WEBSERVER 1`, `FEATURE_HW_VALIDATION 1`, `FEATURE_PSRAM -1`, `HTTP_PORT 80`, `RESIZE_BILINEAR 1`, `START_IN_CONTINUOUS 0`, `ENTROPY_WARN_THRESHOLD 70`), con comentarios de una línea por opción.

- [ ] **Step 4: CMake** — en `firmware/CMakeLists.txt`, después de `project(...)`:

```cmake
# Selección de placa: idf.py -DBOARD=<esp32|esp32cam|esp32s3|esp32s3cam>
if(NOT DEFINED BOARD)
    set(BOARD "esp32cam")
endif()
set(VALID_BOARDS esp32 esp32cam esp32s3 esp32s3cam)
if(NOT BOARD IN_LIST VALID_BOARDS)
    message(FATAL_ERROR "BOARD='${BOARD}' inválido. Válidos: ${VALID_BOARDS}")
endif()
string(TOUPPER "${BOARD}" BOARD_UPPER)
idf_build_set_property(COMPILE_DEFINITIONS "-DBOARD_${BOARD_UPPER}=1" APPEND)
```

- [ ] **Step 5: Compilar** (nada usa aún los headers; valida CMake y que el default funcione):
`bash -c '. ~/esp/esp-idf/export.sh >/dev/null 2>&1 && cd firmware && idf.py -B build_esp32cam -DBOARD=esp32cam build'` → BUILD OK. Y el rechazo: `idf.py -B /tmp/bxx -DBOARD=banana reconfigure` → FATAL_ERROR con la lista.

- [ ] **Step 6: Commit**
```bash
git commit -m "feat(esp32-cam): boards/ + firmware_config.h + selección por -DBOARD" \
  -- firmware/main/boards firmware/main/board.h firmware/main/firmware_config.h \
     firmware/main/firmware_config_check.h firmware/CMakeLists.txt firmware/main/CMakeLists.txt
```
(main/CMakeLists.txt solo si hizo falta tocarlo.)

---

### Task 2: Migrar el firmware de Kconfig a board.h/firmware_config.h

**Files:**
- Modify: `firmware/main/app_main.c`, `firmware/main/camera.c`, `firmware/main/wifi_manager.c`, `firmware/main/http_stream_server.c`, `firmware/main/result_format.c`, `firmware/main/capture_state.c`
- Delete: `firmware/main/Kconfig.projbuild`
- Test: build para `esp32cam`.

**Interfaces:**
- Consumes: macros de Task 1 (`#include "board.h"`).
- Produces: firmware sin `CONFIG_ESP32CAM_*` (salvo MQTT, que queda gated y muerto). Mapa de reemplazos:

| Antes | Después |
|---|---|
| `#ifdef CONFIG_ESP32CAM_CAMERA_ENABLED` | `#if BOARD_HAS_CAMERA` |
| `#ifdef CONFIG_ESP32CAM_WIFI_ENABLED` | `#if FEATURE_WIFI` |
| `#ifdef CONFIG_ESP32CAM_HTTP_STREAM_ENABLED` | `#if FEATURE_WEBSERVER` |
| `CONFIG_ESP32CAM_HTTP_STREAM_PORT` | `HTTP_PORT` |
| `#ifdef CONFIG_ESP32CAM_RESIZE_BILINEAR` (app_main) | `RESIZE_BILINEAR` como valor: pasar `RESIZE_BILINEAR ? true : false` |
| `CONFIG_ESP32CAM_ENTROPY_WARN_THRESHOLD` | `ENTROPY_WARN_THRESHOLD` |
| `CAM_PIN_*` y `20000000` en camera.c | `BOARD_CAM_PIN_*`, `BOARD_CAM_XCLK_FREQ_HZ` |
| `MALLOC_CAP_SPIRAM` (capture_state.c, synth frame) | `EFFECTIVE_PSRAM ? MALLOC_CAP_SPIRAM : MALLOC_CAP_8BIT` |
| `.fb_location = CAMERA_FB_IN_PSRAM` (camera.c) | `EFFECTIVE_PSRAM ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM` |

- [ ] **Step 1:** Agregar `#include "board.h"` a los 6 `.c` y aplicar el mapa completo (grep `CONFIG_ESP32CAM` debe dejar SOLO hits de MQTT en `esp32cam_mqtt.c` y los `#ifdef CONFIG_ESP32CAM_MQTT_ENABLED` de app_main). En `camera.c`, todo el archivo se envuelve en `#if BOARD_HAS_CAMERA` (con stubs `ESP_FAIL`/no-op en `#else` para `camera_init/capture/capture_jpeg/return`, así los callers no necesitan más ifdefs que los que ya tienen).
- [ ] **Step 2:** `capture_state_init()`: si `START_IN_CONTINUOUS` → `s_status.mode = CAPTURE_MODE_CONTINUOUS`.
- [ ] **Step 3:** `run_self_test()` en app_main: envolver el include y el cuerpo con `#if FEATURE_HW_VALIDATION` — con la feature en 0, la tecla `t` imprime `"self-test deshabilitado por config (FEATURE_HW_VALIDATION=0)"` aunque exista `test_images.h`.
- [ ] **Step 4:** Borrar `firmware/main/Kconfig.projbuild` (git rm). El menú desaparece; MQTT queda sin símbolo → compilado out (verificar que el binario linkea sin él).
- [ ] **Step 5:** Build esp32cam OK + `grep -rn "CONFIG_ESP32CAM" firmware/main/ | grep -v MQTT` vacío.
- [ ] **Step 6: Commit**
```bash
git commit -m "refactor(esp32-cam): firmware configurado por board.h/firmware_config.h (chau Kconfig)" \
  -- firmware/main/app_main.c firmware/main/camera.c firmware/main/wifi_manager.c \
     firmware/main/http_stream_server.c firmware/main/result_format.c \
     firmware/main/capture_state.c firmware/main/Kconfig.projbuild
```

---

### Task 3: Imagen fija embebida para placas sin cámara

**Files:**
- Create: `gen_static_frame.py`, `firmware/main/static_frame.h` (generado, TRACKEADO — no depende del modelo, es una imagen fija)
- Modify: `firmware/main/app_main.c`
- Test: `tests/test_static_frame.py` (naming/format del generador) + build.

**Interfaces:**
- Consumes: `get_dataloaders`, `STL10_CLASSES` de `esp32cam_utils`; formato RGB565 del firmware (`preprocess_px_rgb565_to_rgb888` en `preprocess.c` define el layout: `RRRRRGGG GGGBBBBB`).
- Produces: `static_frame.h` con `STATIC_FRAME_W`, `STATIC_FRAME_H`, `STATIC_FRAME_CLASS` (string), `static const uint16_t STATIC_FRAME[W*H]`; `gen_static_frame.py --class-name <stl10-class> --size 320x240 --output firmware/main/static_frame.h` (default: primera imagen de "car" del test set a 96px reescalada a 320×240... NO: emitir a la resolución nativa deseada 320×240 partiendo de la imagen de 96×96 con `torchvision.transforms.Resize`).

- [ ] **Step 1: Test del conversor RGB565** (`tests/test_static_frame.py`, patrón `importlib` de `tests/test_finetune_naming.py` con `SCRIPT_PATH = ... / "gen_static_frame.py"`):

```python
def test_rgb888_to_rgb565_packing():
    mod = _load_module()
    assert mod.rgb888_to_rgb565(255, 255, 255) == 0xFFFF
    assert mod.rgb888_to_rgb565(0, 0, 0) == 0x0000
    assert mod.rgb888_to_rgb565(255, 0, 0) == 0xF800
    assert mod.rgb888_to_rgb565(0, 255, 0) == 0x07E0
    assert mod.rgb888_to_rgb565(0, 0, 255) == 0x001F
```

- [ ] **Step 2:** correr → FAIL (no existe). **Step 3:** implementar `gen_static_frame.py`: carga STL-10 test a 96px, toma la primera imagen de la clase pedida, `Resize((240, 320))`, convierte float[0,1]→uint8→RGB565 con `rgb888_to_rgb565(r,g,b) = ((r&0xF8)<<8)|((g&0xFC)<<3)|(b>>3)`, emite el header con las macros de arriba. **Step 4:** test PASS; generar el header real (`uv run python gen_static_frame.py --class-name car`) y verificar tamaño ~150 KB.
- [ ] **Step 5:** `app_main.c`: reemplazar el frame sintético (bloque `#ifndef CONFIG_ESP32CAM_CAMERA_ENABLED` ya migrado a `#if !BOARD_HAS_CAMERA`) por `#include "static_frame.h"`; `run_pipeline` en el camino sin cámara usa `STATIC_FRAME/STATIC_FRAME_W/H`, reporta `cap_ms = 0`, y el JPEG para la web se encodea desde el frame fijo la primera vez (fmt2jpg funciona igual). Eliminar `synth_frame_init` y `g_synth_fb`.
- [ ] **Step 6:** Build esp32cam OK (no usa el camino) y build **esp32**: `idf.py -B build_esp32 -DBOARD=esp32 build` (primera vez: `idf.py -B build_esp32 set-target esp32` si hace falta) → BUILD OK.
- [ ] **Step 7: Commit**
```bash
git commit -m "feat(esp32-cam): imagen STL-10 fija embebida para placas sin cámara" \
  -- gen_static_frame.py tests/test_static_frame.py firmware/main/static_frame.h firmware/main/app_main.c
```

---

### Task 4: `script05-build.py --board` (tabla, set-target, build dirs, esptool chip)

**Files:**
- Modify: `script05-build.py`, `tests/test_script05_build.py`
- Modify: `pipeline_runner.py` SOLO si algún test lo exige (no debería: `board` viaja como param genérico → `--board`).

**Interfaces:**
- Consumes: convención `-DBOARD=` y build dirs de Task 1.
- Produces:

```python
BOARDS = {
    "esp32":      {"chip": "esp32",   "espdl_target": "c"},
    "esp32cam":   {"chip": "esp32",   "espdl_target": "c"},
    "esp32s3":    {"chip": "esp32s3", "espdl_target": "esp32s3"},
    "esp32s3cam": {"chip": "esp32s3", "espdl_target": "esp32s3"},
}
def board_build_dir(board: str) -> str: return f"build_{board}"
```
- CLI: `--board` (choices = BOARDS, default `esp32cam`); `--target` ELIMINADO (todas las validaciones de sidecar usan `BOARDS[board]["espdl_target"]`).

- [ ] **Step 1: Tests que fallan** (agregar a `tests/test_script05_build.py`):

```python
def test_parser_has_board_and_no_target():
    mod = _load_module()
    args = mod.build_parser().parse_args([])
    assert args.board == "esp32cam"
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--target", "c"])
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--board", "banana"])

def test_board_table_derivations():
    mod = _load_module()
    assert mod.BOARDS["esp32s3cam"] == {"chip": "esp32s3", "espdl_target": "esp32s3"}
    assert mod.BOARDS["esp32"]["espdl_target"] == "c"
    assert mod.board_build_dir("esp32s3") == "build_esp32s3"
```

- [ ] **Step 2:** verificar FAIL. **Step 3:** implementar:
  - `build`: `install_model_data(..., BOARDS[board]["espdl_target"])`; comando `["idf.py", "-B", board_build_dir(board), f"-DBOARD={board}", "build"]`. Antes, si `firmware_dir/build_<board>/` no existe o su `sdkconfig` no contiene `CONFIG_IDF_TARGET="<chip>"`: correr `["idf.py", "-B", bdir, f"-DBOARD={board}", "set-target", chip]`.
  - `flash`: `_run_flash(firmware_dir, port, board)` — `build_dir = firmware_dir / board_build_dir(board)`, `esptool.py --chip {BOARDS[board]['chip']}` (elimina el hardcodeo `esp32`).
  - `monitor`: `["idf.py", "-B", board_build_dir(board), "-p", port, "monitor"]`.
  - `check_installed_target` / preflight: target esperado = derivado del board.
  - `check_psram_enabled`: nueva firma `check_psram_enabled(firmware_dir, board)` — parsear `firmware/main/firmware_config.h` (`FEATURE_PSRAM`, regex `#define\s+FEATURE_PSRAM\s+\(?\s*(-?\d+)`) y `firmware/main/boards/board_<board>.h` (`BOARD_HAS_PSRAM`); si el efectivo es 0 → skip (return); si 1 → chequeo actual sobre `build_<board>/sdkconfig` y `sdkconfig.defaults*`.
- [ ] **Step 4:** actualizar tests existentes que usaban `--target` o el build dir plano `build/` (re-expresarlos con `--board`; conservar la intención de cada uno). Suite completa PASS.
- [ ] **Step 5: Commit**
```bash
git commit -m "feat(esp32-cam): script05 --board (chip/target/builddir derivados de tabla)" \
  -- script05-build.py tests/test_script05_build.py
```

---

### Task 5: `pipelines.yaml` con `board:` + flows nuevos

**Files:**
- Modify: `pipelines.yaml`, `tests/test_main_cli.py`, `tests/test_pipeline_runner.py` (solo si referencian lo cambiado)

**Interfaces:**
- Consumes: `--board` de Task 4 (el runner ya convierte `board: x` → `--board x`).
- Produces flows:
  - Stages `build`/`flash`/`monitor` de `fast_debug_esp32`, `good_model_esp32`, `small_model_esp32` y los genéricos `flash`/`monitor`: agregan `board: esp32cam` (explícito aunque sea el default) y los `build` PIERDEN `target:`/`firmware_dir:` si los tuvieran.
  - `benchmark_esp32`: `extends: fast_debug` + build/flash/monitor con `board: esp32` (placa sin cámara: mide preprocess+inferencia sobre la imagen fija).
  - `benchmark_esp32s3`: `extends: fast_debug` + build con `board: esp32s3` (build-only: sin HW para flashear).
  - `fast_debug_esp32s3cam`: `extends: fast_debug_base` + quantize `target: esp32s3` `firmware_dir: firmware/main`... **NO**: con el proyecto unificado el quantize S3 deja el header en `models/` como siempre y el build lo instala con target derivado del board. Stages: quantize (`target: esp32s3`, sin firmware_dir) + build (`board: esp32s3cam`, `source: ...`). Build-only.
  - Los flows viejos `fast_debug_esp32s3` y `good_model_esp32s3` (que apuntan a `../esp32-s3-cam/`) **NO SE TOCAN** — nota en el YAML: "legacy, proyecto externo; migrar cuando se valide esp32s3cam en HW".
- [ ] **Step 1:** editar YAML según lo anterior. **Step 2:** dry-run de todos los flows (loop como en `tests`/CLI) — los build/flash/monitor deben mostrar `--board <x>` y ninguno `--target`. **Step 3:** suite completa PASS (ajustar asserts de test_main_cli si miran argv). **Step 4: Commit**
```bash
git commit -m "feat(esp32-cam): flows con board: + benchmark_esp32{,s3} + esp32s3cam build-only" \
  -- pipelines.yaml tests/test_main_cli.py tests/test_pipeline_runner.py
```

---

### Task 6: Matriz de builds + validación en HW + docs

**Files:**
- Create: `tools/build_matrix.sh`
- Modify: `CLAUDE.md`, `README.md`
- Test: la matriz misma + self-test en placa (manual, coordinado con el usuario).

- [ ] **Step 1: `tools/build_matrix.sh`** — compila las combinaciones clave y falla al primer error:

```bash
#!/usr/bin/env bash
# Smoke: el firmware compila para todas las placas y combos clave de features.
# Uso: . ~/esp/esp-idf/export.sh && tools/build_matrix.sh
set -e
cd "$(dirname "$0")/../firmware"
for b in esp32 esp32cam esp32s3 esp32s3cam; do
    echo "=== BOARD=$b ==="
    chip=esp32; case "$b" in esp32s3*) chip=esp32s3;; esac
    [ -f "build_$b/sdkconfig" ] || idf.py -B "build_$b" "-DBOARD=$b" set-target "$chip"
    idf.py -B "build_$b" "-DBOARD=$b" build
done
echo "MATRIZ OK"
```

(Los combos de features — webserver off, wifi solo, hw-validation off — se prueban una vez editando `firmware_config.h` en el paso 2 y NO quedan en el script: la matriz corre con la config trackeada.)

- [ ] **Step 2: combos de features (verificación manual, revertir al terminar):** con board `esp32cam`, compilar 3 variantes editando `firmware_config.h`: (a) `FEATURE_WEBSERVER 0` (wifi solo), (b) `FEATURE_WIFI 0` + `FEATURE_WEBSERVER 0` (ninguno), (c) `FEATURE_HW_VALIDATION 0`. Y el caso de error: `FEATURE_WIFI 0` + `FEATURE_WEBSERVER 1` debe cortar con el `#error` de la Task 1. Dejar `firmware_config.h` como estaba (git diff limpio).
- [ ] **Step 3: correr la matriz completa** — 4 builds OK (los S3 quedan "compila", documentado).
- [ ] **Step 4: HW real:** flashear `esp32cam` desde el flow (`uv run main.py run flash` no — usar build+flash con board) y correr self-test `t` → debe seguir 10/10 con dlogits=0. Este paso lo coordina el controlador con el usuario (BOOT+RST manual).
- [ ] **Step 5: Docs:** CLAUDE.md — sección nueva "Placas y configuración" (tabla BOARDS, board.h/firmware_config.h/check, credenciales trackeadas por decisión del usuario, PSRAM override, matriz con tools/build_matrix.sh); actualizar la sección firmware (Kconfig eliminado, `menuconfig` de main.py queda solo para config IDF pura) y "Qué no hacer" (+ "no incluir boards/*.h ni firmware_config.h directo: siempre board.h"). README — actualizar sección firmware + flows nuevos.
- [ ] **Step 6: Commit**
```bash
git commit -m "feat(esp32-cam): matriz de builds multi-placa + docs de configuración" \
  -- tools/build_matrix.sh CLAUDE.md README.md
```

# Diseño: firmware multi-placa + configuración por header

Fecha: 2026-08-13
Directorio: `examples/esp32-cam/` (a futuro se renombrará a `esp32-pipelines`
o similar; fuera del alcance de este spec)

## Objetivo

Un **único proyecto firmware** (`firmware/`) parametrizado para 4 placas —
`esp32`, `esp32cam`, `esp32s3`, `esp32s3cam` (extensible) — con las features
configurables en un header C trackeado, sin Kconfig propio.

## Decisiones tomadas (con el usuario)

- **Un solo proyecto** en `examples/esp32-cam/firmware/`. NO se toca
  `examples/esp32/`, `examples/esp32-s3-cam/` ni `firmware_test_webserver/`.
- **Configuración por `.h` con `#define`s** (estilo `wifi_creds.h`), no
  Kconfig. Las credenciales WiFi van en ese header, **trackeado en el repo**
  (decisión explícita del usuario: no le importa exponerlas).
- La **placa** se selecciona en build-time desde el pipeline
  (`idf.py -DBOARD=...`), no en el header: los flows de `pipelines.yaml`
  buildean para placas distintas sin editar archivos.
- **WiFi es una feature separada del webserver**. Combinaciones válidas:
  wifi+webserver, wifi solo, ninguno. Webserver sin wifi → error de compilación.
- **PSRAM configurable por el usuario** más allá del default de la placa
  (la ESP32-S3-CAM del usuario tiene la PSRAM rota → forzar 0).
- MQTT queda **fuera** de la config nueva (código presente, compilado out).
- Placas sin cámara: inferencia sobre una **imagen fija embebida** (solo
  tiempos de preprocess+inferencia importan).
- Backlog ya asentado en README: validación end-to-end con cámara pendiente.

## 1. Selección de placa

### `firmware/main/boards/`

Un header por placa con lo **intrínseco** (no configurable por el usuario):

```c
// boards/board_esp32cam.h (ejemplo)
#pragma once
#define BOARD_NAME        "esp32cam"
#define BOARD_HAS_CAMERA  1
#define BOARD_HAS_PSRAM   1
// pines de cámara AI-Thinker (solo si BOARD_HAS_CAMERA)
#define BOARD_CAM_PIN_PWDN 32
// ... resto de pines ...
```

- `board_esp32.h` / `board_esp32s3.h`: `BOARD_HAS_CAMERA 0`, sin pines.
- `board_esp32s3cam.h`: pines del proyecto S3 existente, `BOARD_HAS_PSRAM 1`
  (el override del usuario la apaga vía `FEATURE_PSRAM 0`).
- Un header genérico `board.h` incluye el header de la placa según el define
  `BOARD_<NAME>` que inyecta CMake, y valida que exactamente uno esté activo.

### CMake / build

- `idf.py -DBOARD=esp32cam` (default `esp32cam` si no se pasa) → CMake agrega
  `-DBOARD_ESP32CAM=1` como compile definition. Placa desconocida → error de
  configure con la lista de placas válidas.
- Pines de cámara hoy hardcodeados en `camera.c` migran a los board headers.

### `script05-build.py`

- Nueva flag `--board {esp32,esp32cam,esp32s3,esp32s3cam}` (default
  `esp32cam`). Deriva TODO de una tabla única:

| board | chip IDF (`set-target`) | target `.espdl` | build dir |
|---|---|---|---|
| esp32 | esp32 | c | `build_esp32/` |
| esp32cam | esp32 | c | `build_esp32cam/` |
| esp32s3 | esp32s3 | esp32s3 | `build_esp32s3/` |
| esp32s3cam | esp32s3 | esp32s3 | `build_esp32s3cam/` |

- La flag `--target` actual **desaparece** (queda derivada del board). La
  validación del sidecar (`target` del `.espdl` vs chip) se mantiene idéntica,
  contra el target derivado.
- Build dir por placa (`idf.py -B build_<board> -DBOARD=<board>`): cambiar de
  placa no invalida el build de otra. `flash` y `monitor` también reciben
  `--board` para elegir el build dir y el chip de esptool (`--chip esp32|esp32s3`)
  — esto resuelve de paso el hardcodeo de `--chip esp32` en `_run_flash`.
- `set-target`: si `build_<board>/` no existe o el chip no coincide, correr
  `idf.py -B build_<board> set-target <chip>` antes del build. sdkconfig por
  build dir (`SDKCONFIG=build_<board>/sdkconfig`), partiendo de
  `sdkconfig.defaults` + `sdkconfig.defaults.<chip>` (convención IDF).
- `pipelines.yaml`: los stages `build`/`flash`/`monitor` llevan `board:` en
  vez de `target:`/`firmware_dir` para las placas de este proyecto. El flow
  `*_esp32s3` de `../esp32-s3-cam/` existente se migra a `board: esp32s3cam`
  cuando ese hardware se pruebe; mientras tanto se agrega un flow paralelo y
  el viejo no se toca (esa carpeta es de otra sesión).

## 2. Configuración de features: `firmware/main/firmware_config.h`

Trackeado en el repo. Contenido inicial:

```c
#pragma once
/* WiFi (credenciales trackeadas a propósito; "" = red abierta) */
#define FEATURE_WIFI           1
#define WIFI_SSID              "TeleCentro-fecf"
#define WIFI_PASSWORD          "sinlugaradudas"

/* Features */
#define FEATURE_WEBSERVER      1   /* requiere FEATURE_WIFI */
#define FEATURE_HW_VALIDATION  1   /* golden images + tecla 't' */
#define FEATURE_PSRAM          -1  /* -1 = default de la placa; 0/1 fuerza */

/* Parámetros */
#define HTTP_PORT              80
#define RESIZE_BILINEAR        1
#define START_IN_CONTINUOUS    0
#define ENTROPY_WARN_THRESHOLD 70  /* % */
```

### Reglas (en `firmware_config_check.h`, incluido por `board.h`)

- `FEATURE_WEBSERVER && !FEATURE_WIFI` → `#error "webserver requiere wifi"`.
- `!BOARD_HAS_CAMERA` → la captura por cámara se compila out (manda la placa;
  no hay flag de usuario para la cámara).
- `FEATURE_PSRAM`: -1 → usa `BOARD_HAS_PSRAM`; 0/1 fuerza. El valor efectivo
  (`EFFECTIVE_PSRAM`) decide dónde viven los buffers grandes (JPEG de
  capture_state, frame sintético) y si el preflight de PSRAM de script05
  aplica (ver §4).
- `FEATURE_HW_VALIDATION 0` → no se incluye `test_images.h` (aunque exista) y
  la tecla `t` responde "self-test deshabilitado por config".

### Qué reemplaza

- El menú Kconfig `ESP32CAM_*` se elimina (`Kconfig.projbuild` queda solo si
  IDF lo exige vacío). Migración: `CAMERA_ENABLED`→`BOARD_HAS_CAMERA`,
  `WIFI_ENABLED`→`FEATURE_WIFI`, `HTTP_STREAM_ENABLED`→`FEATURE_WEBSERVER`,
  `HTTP_STREAM_PORT`→`HTTP_PORT`, `RESIZE_BILINEAR` y
  `ENTROPY_WARN_THRESHOLD` → header. `MQTT_*`: el código de
  `esp32cam_mqtt.c` queda compilado out sin entrada en la config nueva.
- `firmware/` deja de incluir el `wifi_creds.h` compartido (que queda solo
  para `firmware_test_webserver/`, intocado).

## 3. Placas sin cámara (esp32, esp32-s3)

- Se embebe una imagen fija en RGB565 (`static_frame.h`, generada una vez a
  partir de una imagen del test set de STL-10 con un script/función junto a
  `gen_test_images.py`) para que el trigger corra el pipeline **real** de
  preprocess+inferencia sobre datos con sentido.
- Reporte serie/web: `t_capture_ms = 0` (u omitido), se informan preprocess,
  inferencia y total. La web (si está) muestra la imagen fija como preview.
- El frame sintético actual (gradiente) se reemplaza por esta imagen fija.

## 4. Interacción con el pipeline Python

- `check_psram_enabled` en script05: pasa a depender del board+config —
  para `FEATURE_PSRAM` efectivo 0 no exige `CONFIG_SPIRAM` en sdkconfig.
  (Lectura simple: script05 parsea `firmware_config.h` y el board header con
  el mismo regex-parser que ya usa para `model_data.h`.)
- El self-test/golden images no cambian; `FEATURE_HW_VALIDATION` solo decide
  si se embeben.
- Tests Python: tabla board→(chip, target, builddir) y derivaciones; parser
  de `firmware_config.h`; validación de sidecar contra target derivado.

## Verificación

- `idf.py -DBOARD=<b>` debe compilar para las 4 placas × combinaciones clave
  de features (webserver on/off, wifi solo, hw-validation off) — matriz de
  builds como smoke test (script o instrucciones documentadas).
- En hardware real: solo `esp32cam` verificable ahora (self-test 10/10 debe
  seguir pasando). Las S3 quedan "compila OK" hasta tener la placa — es el
  paso siguiente planificado (ítems 7/8 del backlog de la sesión).
- `uv run pytest tests/` verde.

## Fuera de alcance

- Rename de `examples/esp32-cam/` → `esp32-pipelines`.
- Migrar/retirar `examples/esp32-s3-cam/` y `firmware_test_webserver/`.
- MQTT en la config nueva.
- Flasheo/validación en hardware S3.

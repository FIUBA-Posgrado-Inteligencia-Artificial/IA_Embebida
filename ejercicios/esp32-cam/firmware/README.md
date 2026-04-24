# Firmware ESP32-CAM — Pipeline de inferencia

Firmware ESP-IDF 5.3+ para el AI-Thinker ESP32-CAM. Captura una imagen, corre inferencia ESP-DL sobre el modelo en `main/model_data.h` (generado por `script03-quantize.py`) y emite el resultado por serie y opcionalmente por MQTT.

## Requisitos

- ESP-IDF ≥ 5.3 (https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html)
- Hardware: AI-Thinker ESP32-CAM + adaptador USB (ESP32-CAM-MB o FTDI)
- CMake ≥ 3.16 (para tests de host)
- `model_data.h` generado previamente por el pipeline Python:
  ```bash
  cd examples/esp32-cam
  uv run python script03-quantize.py --source depthwise_gray32 --mode ptq
  ```
  Si no corres esto, el firmware usa un modelo vacío (modo dummy: probs uniformes, entropía = 1.0).

## Build y flash

```bash
cd examples/esp32-cam/firmware
idf.py set-target esp32
idf.py menuconfig        # opcional: ajustar Component config → ESP32-CAM Demo
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

Para salir del monitor: `Ctrl-]`.

## Verificación manual

### 1. Modo serial-only (WiFi y MQTT deshabilitados)

1. `idf.py menuconfig` → `Component config → ESP32-CAM Demo` → dejar `WiFi enabled = n`.
2. `idf.py flash monitor`.
3. Esperar el mensaje `ready — serial: any byte / MQTT: (disabled)`.
4. En la consola serie, presionar cualquier tecla (ej. enter).
5. Ver una línea de texto y una de JSON con el top1, confidence, entropy y timing.

### 2. Con WiFi

1. `menuconfig` → habilitar `WiFi enabled`, setear `WiFi SSID` y `WiFi password`.
2. Rebuild + flash.
3. En el monitor debería aparecer `[WIFI] connected, IP=…`.
4. Trigger serial sigue funcionando.

### 3. Con MQTT

1. `menuconfig` → habilitar `MQTT enabled`, setear `MQTT broker URI`.
2. Rebuild + flash.
3. Esperar `[MQTT] connected, subscribing to esp32cam/capture`.
4. Desde otra máquina con el broker alcanzable:
   ```bash
   mosquitto_pub -h <broker> -t esp32cam/capture -m ""
   mosquitto_sub -h <broker> -t esp32cam/result
   ```
5. Cada publish en `capture` dispara una inferencia y publica el JSON en `result`.

## Tests de host (preprocessing)

```bash
cd examples/esp32-cam/firmware/tests/host
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Todos los tests deberían pasar (>20 casos sobre `preprocess.c`). No requiere ESP-IDF ni hardware.

## Opciones de menuconfig

| Opción | Default | Descripción |
|---|---|---|
| `WiFi enabled` | n | Activa la conexión WiFi |
| `WiFi SSID` / `WiFi password` | `MySSID` / `MyPassword` | Credenciales (solo si WiFi on) |
| `MQTT enabled` | n | Activa el cliente MQTT (requiere WiFi) |
| `MQTT broker URI` | `mqtt://192.168.1.100:1883` | URL del broker |
| `MQTT username` / `MQTT password` | vacíos | Auth opcional |
| `MQTT subscribe/publish topic` | `esp32cam/capture` / `esp32cam/result` | Topics |
| `Resize bilinear` | y | Bilineal vs nearest neighbor |
| `Entropy warn threshold (0-100)` | 70 | Umbral de anotación low confidence |

## Troubleshooting

- **`camera init failed`**: verificar que el ESP32-CAM tiene PSRAM soldada y que `sdkconfig` tiene `CONFIG_ESP32_SPIRAM_SUPPORT=y`. Probar `idf.py menuconfig → Component config → ESP PSRAM`.
- **`[WIFI] disconnected, retrying...`**: SSID/password incorrectos o red fuera de alcance. Revisar `menuconfig` y confirmar que el AP es 2.4 GHz (ESP32 no soporta 5 GHz).
- **`[MQTT] not connected, drop result`**: el broker no está alcanzable desde el device. Probar `ping <broker>` desde otra máquina en la misma red.
- **Modelo dummy siempre**: significa que `model_data_len == 0`. Regenerar con `script03-quantize.py` con un modelo válido y reflashear.
- **Build falla por API de ESP-DL**: abrir `managed_components/espressif__esp-dl/examples/` y adaptar `inference.cpp` al API de la versión pineada.

## Estructura interna

```
firmware/
├── CMakeLists.txt
├── idf_component.yml         # Deps: esp-dl, esp32-camera
├── partitions.csv
├── sdkconfig.defaults
├── main/
│   ├── app_main.c            # Init + main loop (queue-based triggers)
│   ├── camera.c/.h           # OV2640 wrapper (pines AI-Thinker)
│   ├── preprocess.c/.h       # RGB565 → resize → gray → int8 (pure C)
│   ├── inference.cpp/.h      # ESP-DL v2 + softmax + entropía
│   ├── result_format.c/.h    # Texto + JSON builder
│   ├── serial_trigger.c/.h   # UART0 trigger task
│   ├── wifi_manager.c/.h     # [#ifdef CONFIG_WIFI] STA + retry
│   ├── esp32cam_mqtt.c/.h    # [#ifdef CONFIG_MQTT] client + pub/sub
│   ├── trigger.h             # Enum compartido de trigger
│   ├── Kconfig.projbuild     # Opciones de menuconfig
│   └── model_data.h          # Generado por script03-quantize.py
└── tests/host/               # Unity + tests de preprocess (no requiere HW)
```

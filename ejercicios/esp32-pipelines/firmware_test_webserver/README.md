# `firmware_test_webserver`

Minimal ESP-IDF firmware for the AI-Thinker ESP32-CAM that serves a live
MJPEG camera stream over HTTP. Intended as an isolation test: if this
works, WiFi + camera + HTTP are healthy, so any problem in `firmware/`
is in the parts it adds on top (inference, MQTT).

## Prerequisites

- ESP-IDF ≥ 5.3 installed at `~/esp/esp-idf/`.
- AI-Thinker ESP32-CAM connected via the ESP32-CAM-MB programmer board
  (appears as `/dev/ttyUSB0`).

## Build and flash

```bash
cd examples/esp32-cam/firmware_test_webserver

# First time only: create the gitignored wifi_creds.h shared with firmware/.
cp ../wifi_creds.example.h ../wifi_creds.h
$EDITOR ../wifi_creds.h        # fill in SSID + password

. ~/esp/esp-idf/export.sh
idf.py set-target esp32
idf.py -p /dev/ttyUSB0 flash monitor
```

The AI-Thinker ESP32-CAM-MB programmer has no auto-reset circuit. To
enter bootloader mode for flashing:

1. Hold the **BOOT** button.
2. Press and release **RST**.
3. Release **BOOT**.

Keep `idf.py flash` running while you do the BOOT+RST sequence.

## View the stream

Watch the serial monitor for:

```
I (xxxx) test_webserver: camera init OK — VGA JPEG
I (xxxx) test_webserver: connecting to "..."
I (xxxx) test_webserver: connected, IP=192.168.X.Y
I (xxxx) test_webserver: HTTP server started on port 80 — open http://<IP>/
```

Open `http://192.168.X.Y/` in a browser — you should see the live feed.

## Open networks

Leave `WIFI_PASSWORD` empty (`""`) to join an open network — the code
switches `threshold.authmode` to `WIFI_AUTH_OPEN` automatically (with
the default `WPA2_PSK` threshold the driver silently discards open APs
and never associates).

## Sensor support: OV2640 and OV3660

The firmware auto-detects the sensor via the esp32-camera driver;
`CONFIG_OV2640_SUPPORT` and `CONFIG_OV3660_SUPPORT` are both enabled,
so swapping camera modules requires no rebuild.

## Troubleshooting (field notes, 2026-08-07)

Symptoms actually observed on an AI-Thinker ESP32-CAM with a USB-C
ESP32-CAM-MB, and what they turned out to mean:

- **`NO-SOI - JPEG start marker missing` flood, `/snap` returns 500,
  stream shows nothing** — the sensor never emits a valid JPEG. If a
  diagnostic build with `PIXFORMAT_RGB565` *does* deliver frames, the
  parallel data bus and ribbon are fine and the sensor's internal JPEG
  encoder is dead. That exact failure happened here with an OV2640
  (raw frames OK, JPEG never); swapping to an OV3660 fixed it
  instantly with the same firmware. Not power, not software.
- **Intermittent `SCCB_Write Failed` on random registers during init**
  — happens on cold boot at both 10 and 20 MHz XCLK; the 5-attempt
  retry loop in `camera_init()` rides it out. At 20 MHz it got worse
  (PID sometimes read corrupt: "Detected camera not supported"), which
  is why XCLK stays at 10 MHz.
- **Board doesn't boot after `idf.py flash` finishes** — the USB-C MB's
  auto-reset is unreliable: it sometimes enters the bootloader by
  itself but usually needs manual BOOT+RST to flash, and a manual RST
  afterwards to start the app. Flashing with
  `--connect-attempts 120` leaves a ~4 min window to do the button
  dance.
- **`disconnected, retrying in 2s` loop** — check the SSID actually
  exists where you are (scan with `nmcli dev wifi`); stale
  `wifi_creds.h` from another location looks identical to a real
  failure.

## What this firmware deliberately does NOT have

No inference, no MQTT, no serial trigger, no preprocess, no
`model_data.h`, no Kconfig menu. If you need any of those, use
`firmware/` instead.

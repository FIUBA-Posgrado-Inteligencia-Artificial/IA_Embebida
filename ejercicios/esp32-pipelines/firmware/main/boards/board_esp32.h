#pragma once
#define BOARD_NAME        "esp32"
#define BOARD_HAS_CAMERA  0
/* Los devkit ESP32 pelados típicamente no traen PSRAM; si el tuyo trae,
 * forzá FEATURE_PSRAM 1 en firmware_config.h. */
#define BOARD_HAS_PSRAM   0

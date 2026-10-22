#pragma once
#define BOARD_NAME        "esp32s3cam"
#define BOARD_HAS_CAMERA  1
/* La PSRAM de esta placa no inicializa (placa relevada) — forzar
 * FEATURE_PSRAM 0 en firmware_config.h. */
#define BOARD_HAS_PSRAM   1
#define BOARD_CAM_XCLK_FREQ_HZ 20000000
#define BOARD_CAM_PIN_PWDN  -1
#define BOARD_CAM_PIN_RESET -1
#define BOARD_CAM_PIN_XCLK  15
#define BOARD_CAM_PIN_SIOD   4
#define BOARD_CAM_PIN_SIOC   5
#define BOARD_CAM_PIN_D7    16
#define BOARD_CAM_PIN_D6    17
#define BOARD_CAM_PIN_D5    18
#define BOARD_CAM_PIN_D4    12
#define BOARD_CAM_PIN_D3    10
#define BOARD_CAM_PIN_D2     8
#define BOARD_CAM_PIN_D1     9
#define BOARD_CAM_PIN_D0    11
#define BOARD_CAM_PIN_VSYNC  6
#define BOARD_CAM_PIN_HREF   7
#define BOARD_CAM_PIN_PCLK  13

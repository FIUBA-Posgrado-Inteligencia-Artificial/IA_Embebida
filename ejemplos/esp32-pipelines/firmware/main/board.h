/* board.h — punto de entrada único de configuración. Incluir SIEMPRE este,
 * nunca boards/(placa).h ni firmware_config.h directo. */
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

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

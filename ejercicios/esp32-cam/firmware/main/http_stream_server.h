#pragma once
#include "esp_err.h"

/* Arranca el servidor HTTP en el puerto 80.
 * Endpoints: GET / (página HTML) y GET /stream (MJPEG).
 * Requiere que WiFi esté inicializado (wifi_manager_start llamado antes). */
esp_err_t http_stream_server_start(void);

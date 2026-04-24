#pragma once
#include "esp_err.h"
#include "esp_http_server.h"

/* Arranca el servidor HTTP en el puerto 80.
 * Endpoints: GET / (página HTML) y GET /stream (MJPEG).
 * Requiere que WiFi esté inicializado (wifi_manager_start llamado antes). */
esp_err_t http_stream_server_start(void);

/* Devuelve el handle del servidor HTTP iniciado por http_stream_server_start
 * para que otros módulos (p.ej. ota_handler) puedan registrar URIs adicionales.
 * Retorna NULL si el servidor no está levantado. */
httpd_handle_t http_stream_server_get_handle(void);

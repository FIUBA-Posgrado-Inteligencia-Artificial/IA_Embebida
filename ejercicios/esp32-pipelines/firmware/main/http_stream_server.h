#pragma once
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

/* Arranca el servidor HTTP en el puerto 80.
 * Endpoints: GET / (página HTML), GET /stream (MJPEG), POST /trigger,
 * POST /mode (body continuous=0|1), GET /last.jpg, GET /result.
 * Requiere que WiFi esté inicializado (wifi_manager_start llamado antes).
 * trigger_queue: la misma cola que consumen serial_trigger/mqtt triggers. */
esp_err_t http_stream_server_start(QueueHandle_t trigger_queue);

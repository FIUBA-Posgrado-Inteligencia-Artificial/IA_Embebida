#pragma once
#include "esp_err.h"
#include "esp_http_server.h"

/* Registra POST /ota y GET /ota/info sobre un httpd ya inicializado.
 * El handler de POST acepta la imagen cruda del binario (Content-Type es
 * irrelevante); la escribe en la siguiente partición OTA y reinicia el chip
 * en cuanto esp_ota_set_boot_partition devuelve OK.
 *
 * Pasar un server NULL es no-op (sirve cuando el módulo está compilado
 * pero el servidor HTTP no arrancó). */
esp_err_t ota_handler_register(httpd_handle_t server);

#pragma once
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

/* Crea una task que bloquea en UART0 RX. Cada byte recibido → TRIGGER_SERIAL
 * en la queue. Drena el buffer antes de encolar para no acumular triggers. */
esp_err_t serial_trigger_start(QueueHandle_t trigger_queue);

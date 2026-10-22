#pragma once
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

/* Crea una task que bloquea en UART0 RX. Teclas: c=captura, m=modo,
 * t=self-test, h=ayuda; el resto se ignora. Drena el buffer antes de
 * encolar para no acumular triggers. */
esp_err_t serial_trigger_start(QueueHandle_t trigger_queue);

/* Imprime el menú de opciones de la terminal (también al bootear). */
void serial_trigger_print_help(void);

#include "serial_trigger.h"
#include "trigger.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "freertos/task.h"

static const char *TAG = "serial_trig";
#define UART_NUM  UART_NUM_0

static void serial_task(void *pv)
{
    QueueHandle_t queue = (QueueHandle_t)pv;
    uint8_t byte;
    while (1) {
        int n = uart_read_bytes(UART_NUM, &byte, 1, portMAX_DELAY);
        if (n <= 0) continue;
        /* drena buffer pendiente para consolidar en un único trigger */
        uart_flush_input(UART_NUM);
        trigger_source_t src = TRIGGER_SERIAL;
        xQueueSend(queue, &src, 0);
    }
}

esp_err_t serial_trigger_start(QueueHandle_t trigger_queue)
{
    /* UART0 ya lo inicializó la consola del IDF; sólo instalamos driver RX. */
    esp_err_t err = uart_driver_install(UART_NUM, 256, 0, 0, NULL, 0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "driver install failed: %s", esp_err_to_name(err));
        return err;
    }
    xTaskCreate(serial_task, "serial_trig", 2048, trigger_queue, 5, NULL);
    return ESP_OK;
}

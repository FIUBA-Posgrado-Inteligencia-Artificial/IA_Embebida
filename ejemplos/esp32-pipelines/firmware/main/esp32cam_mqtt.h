#pragma once
#include <stdbool.h>
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

esp_err_t esp32cam_mqtt_start(QueueHandle_t trigger_queue);
esp_err_t esp32cam_mqtt_publish_result(const char *json_payload);
bool      esp32cam_mqtt_is_connected(void);

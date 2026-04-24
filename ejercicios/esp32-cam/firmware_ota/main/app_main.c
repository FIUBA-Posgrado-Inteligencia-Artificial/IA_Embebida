#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "sdkconfig.h"

#include "camera.h"
#include "preprocess.h"
#include "inference.h"
#include "result_format.h"
#include "serial_trigger.h"
#include "wifi_manager.h"
#include "esp32cam_mqtt.h"
#include "http_stream_server.h"
#include "ota_handler.h"
#include "trigger.h"
#include "model_data.h"

static const char *TAG = "app_main";

static int8_t g_input_buf[MODEL_INPUT_W * MODEL_INPUT_H * MODEL_INPUT_CHANNELS];

static void run_pipeline(trigger_source_t src)
{
    camera_fb_t *fb = NULL;
    if (camera_capture(&fb) != ESP_OK || fb == NULL) return;

    uint8_t mean[3] = { MODEL_INPUT_MEAN_0,
#if MODEL_INPUT_CHANNELS == 3
                        MODEL_INPUT_MEAN_1, MODEL_INPUT_MEAN_2
#else
                        0, 0
#endif
                      };
    uint8_t std[3]  = { MODEL_INPUT_STD_0,
#if MODEL_INPUT_CHANNELS == 3
                        MODEL_INPUT_STD_1, MODEL_INPUT_STD_2
#else
                        1, 1
#endif
                      };

    /* fb->height is the configured dimension; the OV2640 DVP timing can deliver
     * fewer lines (always an exact half-buffer period short). Derive the actual
     * height from fb->len so the resize is correct regardless. */
    int actual_w = (int)fb->width;
    int actual_h = (fb->format == PIXFORMAT_RGB565 && actual_w > 0)
                   ? (int)fb->len / (actual_w * 2)
                   : (int)fb->height;

    preprocess_rgb565(
        (const uint16_t *)fb->buf,
        actual_w, actual_h,
        g_input_buf,
        MODEL_INPUT_W, MODEL_INPUT_H, MODEL_INPUT_CHANNELS,
        mean, std,
#ifdef CONFIG_ESP32CAM_RESIZE_BILINEAR
        true
#else
        false
#endif
    );
    camera_return(fb);

    inference_result_t result;
    inference_run(g_input_buf, &result);

    char text[160], json[768];
    result_format_text(&result, text, sizeof text);
    result_format_json(&result, json, sizeof json);
    printf("src=%s %s\n", src == TRIGGER_SERIAL ? "serial" : "mqtt", text);
    printf("%s\n", json);

#ifdef CONFIG_ESP32CAM_MQTT_ENABLED
    esp32cam_mqtt_publish_result(json);
#endif
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    if (camera_init() != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed, rebooting in 3s");
        vTaskDelay(pdMS_TO_TICKS(3000));
        esp_restart();
    }
    inference_init();
    if (inference_is_dummy_mode()) {
        ESP_LOGW(TAG, "running in dummy inference mode");
    }

    QueueHandle_t trigger_queue = xQueueCreate(4, sizeof(trigger_source_t));
    serial_trigger_start(trigger_queue);

#ifdef CONFIG_ESP32CAM_WIFI_ENABLED
    /* Cold boot note: PHY init reliably crashes on the first boot after a
     * full power-cycle (3.3V rail transient during RF activation). The
     * firmware auto-recovers on the next boot (PSRAM caps stay charged,
     * so warm reset succeeds). Net effect: ~2s longer first-boot latency
     * after a power-cycle; zero impact during dev iterations. We don't
     * try to prevent it in software — we just let it happen. */
    wifi_manager_start();
#endif
#ifdef CONFIG_ESP32CAM_MQTT_ENABLED
    esp32cam_mqtt_start(trigger_queue);
#endif
#ifdef CONFIG_ESP32CAM_HTTP_STREAM_ENABLED
    http_stream_server_start();
#ifdef CONFIG_ESP32CAM_OTA_ENABLED
    ota_handler_register(http_stream_server_get_handle());
#endif
#endif

    ESP_LOGI(TAG, "ready — serial: any byte / MQTT: publish to %s",
#ifdef CONFIG_ESP32CAM_MQTT_ENABLED
             CONFIG_ESP32CAM_MQTT_TOPIC_CAPTURE
#else
             "(disabled)"
#endif
    );

    trigger_source_t src;
    while (1) {
        if (xQueueReceive(trigger_queue, &src, portMAX_DELAY) == pdTRUE) {
            run_pipeline(src);
        }
    }
}

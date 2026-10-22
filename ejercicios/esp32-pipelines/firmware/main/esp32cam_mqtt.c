#include "esp32cam_mqtt.h"
#include "sdkconfig.h"

#ifdef CONFIG_ESP32CAM_MQTT_ENABLED

#include <string.h>
#include "trigger.h"
#include "esp_log.h"
#include "mqtt_client.h"   /* el oficial de IDF */

static const char *TAG = "mqtt";
static esp_mqtt_client_handle_t g_client = NULL;
static QueueHandle_t g_trigger_queue = NULL;
static bool g_connected = false;

static void on_event(void *handler_args, esp_event_base_t base,
                     int32_t event_id, void *event_data)
{
    esp_mqtt_event_handle_t e = (esp_mqtt_event_handle_t)event_data;
    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            g_connected = true;
            ESP_LOGI(TAG, "connected, subscribing to %s",
                     CONFIG_ESP32CAM_MQTT_TOPIC_CAPTURE);
            esp_mqtt_client_subscribe(g_client,
                CONFIG_ESP32CAM_MQTT_TOPIC_CAPTURE, 0);
            break;
        case MQTT_EVENT_DISCONNECTED:
            g_connected = false;
            ESP_LOGW(TAG, "disconnected");
            break;
        case MQTT_EVENT_DATA: {
            if (g_trigger_queue && strncmp(e->topic,
                    CONFIG_ESP32CAM_MQTT_TOPIC_CAPTURE,
                    e->topic_len) == 0) {
                trigger_source_t src = TRIGGER_MQTT;
                xQueueSend(g_trigger_queue, &src, 0);
            }
            break;
        }
        default: break;
    }
}

esp_err_t esp32cam_mqtt_start(QueueHandle_t trigger_queue)
{
    g_trigger_queue = trigger_queue;

    esp_mqtt_client_config_t cfg = {
        .broker.address.uri = CONFIG_ESP32CAM_MQTT_BROKER_URI,
    };
    if (strlen(CONFIG_ESP32CAM_MQTT_USERNAME) > 0) {
        cfg.credentials.username = CONFIG_ESP32CAM_MQTT_USERNAME;
        cfg.credentials.authentication.password = CONFIG_ESP32CAM_MQTT_PASSWORD;
    }
    g_client = esp_mqtt_client_init(&cfg);
    if (!g_client) return ESP_FAIL;

    esp_mqtt_client_register_event(g_client, ESP_EVENT_ANY_ID, on_event, NULL);
    return esp_mqtt_client_start(g_client);
}

esp_err_t esp32cam_mqtt_publish_result(const char *json_payload)
{
    if (!g_connected) {
        ESP_LOGW(TAG, "not connected, drop result");
        return ESP_FAIL;
    }
    int msg_id = esp_mqtt_client_publish(g_client,
        CONFIG_ESP32CAM_MQTT_TOPIC_RESULT, json_payload, 0, 0, 0);
    return msg_id >= 0 ? ESP_OK : ESP_FAIL;
}

bool esp32cam_mqtt_is_connected(void) { return g_connected; }

#else  /* MQTT disabled */

esp_err_t esp32cam_mqtt_start(QueueHandle_t q) { (void)q; return ESP_OK; }
esp_err_t esp32cam_mqtt_publish_result(const char *p) { (void)p; return ESP_OK; }
bool      esp32cam_mqtt_is_connected(void) { return false; }

#endif

#include "wifi_manager.h"
#include "sdkconfig.h"

#ifdef CONFIG_ESP32CAM_WIFI_ENABLED

#include <string.h>
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

static const char *TAG = "wifi";
#define BIT_CONNECTED (1 << 0)
static EventGroupHandle_t g_evt;
static bool g_connected = false;
static int g_backoff_s = 1;

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        g_connected = false;
        xEventGroupClearBits(g_evt, BIT_CONNECTED);
        ESP_LOGW(TAG, "disconnected, retrying in %ds", g_backoff_s);
        vTaskDelay(pdMS_TO_TICKS(g_backoff_s * 1000));
        if (g_backoff_s < 10) g_backoff_s = (g_backoff_s == 1) ? 2
                                          : (g_backoff_s == 2) ? 5 : 10;
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "connected, IP=" IPSTR, IP2STR(&e->ip_info.ip));
        g_backoff_s = 1;
        g_connected = true;
        xEventGroupSetBits(g_evt, BIT_CONNECTED);
    }
}

esp_err_t wifi_manager_start(void)
{
    g_evt = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t icfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&icfg));

    esp_event_handler_instance_t h1, h2;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL, &h1));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL, &h2));

    wifi_config_t cfg = { 0 };
    strncpy((char *)cfg.sta.ssid,     CONFIG_ESP32CAM_WIFI_SSID,     sizeof(cfg.sta.ssid));
    strncpy((char *)cfg.sta.password, CONFIG_ESP32CAM_WIFI_PASSWORD, sizeof(cfg.sta.password));
    cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "connecting to \"%s\"", CONFIG_ESP32CAM_WIFI_SSID);
    return ESP_OK;
}

bool wifi_manager_is_connected(void) { return g_connected; }

#else  /* WIFI disabled */

esp_err_t wifi_manager_start(void) { return ESP_OK; }
bool      wifi_manager_is_connected(void) { return false; }

#endif

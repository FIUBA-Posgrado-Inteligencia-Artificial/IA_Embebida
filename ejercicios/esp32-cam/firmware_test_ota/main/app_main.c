/* esp32-cam test_ota firmware — WiFi + HTTP + push OTA, nada más.
 *
 * Endpoints:
 *   GET  /         texto plano con help
 *   GET  /info     JSON con running/boot/update partition
 *   POST /ota      cuerpo = .bin crudo; escribe en ota_next y reinicia
 *
 * Si esto anda y firmware_ota/ no, el problema está en el stack de
 * cámara/esp-dl, no en el pipeline OTA.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_ota_ops.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

static const char *TAG = "test_ota";

#define WIFI_CONNECTED_BIT BIT0
#define OTA_RECV_CHUNK     4096

static EventGroupHandle_t s_wifi_evt;

/* -------------------------------------------------------------------------- */
/* WiFi                                                                       */
/* -------------------------------------------------------------------------- */

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "wifi disconnected — retrying in 2s");
        xEventGroupClearBits(s_wifi_evt, WIFI_CONNECTED_BIT);
        vTaskDelay(pdMS_TO_TICKS(2000));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "got IP: " IPSTR, IP2STR(&e->ip_info.ip));
        xEventGroupSetBits(s_wifi_evt, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init(void)
{
    s_wifi_evt = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t icfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&icfg));

    ESP_ERROR_CHECK(esp_event_handler_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL));

    wifi_config_t wc = { 0 };
    strncpy((char *)wc.sta.ssid,     CONFIG_TEST_OTA_WIFI_SSID,     sizeof(wc.sta.ssid));
    strncpy((char *)wc.sta.password, CONFIG_TEST_OTA_WIFI_PASSWORD, sizeof(wc.sta.password));
    wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "connecting to \"%s\"", CONFIG_TEST_OTA_WIFI_SSID);
}

/* -------------------------------------------------------------------------- */
/* HTTP handlers                                                              */
/* -------------------------------------------------------------------------- */

static esp_err_t root_handler(httpd_req_t *req)
{
    const char *msg =
        "esp32-cam test_ota firmware\n"
        "\n"
        "endpoints:\n"
        "  GET  /         this page\n"
        "  GET  /info     running/boot/update partition (JSON)\n"
        "  POST /ota      upload new app .bin (body = raw binary)\n";
    httpd_resp_set_type(req, "text/plain");
    return httpd_resp_sendstr(req, msg);
}

static esp_err_t info_handler(httpd_req_t *req)
{
    const esp_partition_t *running = esp_ota_get_running_partition();
    const esp_partition_t *boot    = esp_ota_get_boot_partition();
    const esp_partition_t *update  = esp_ota_get_next_update_partition(NULL);

    char resp[384];
    int n = snprintf(resp, sizeof resp,
                     "{"
                     "\"running\":\"%s\","
                     "\"boot\":\"%s\","
                     "\"update_target\":\"%s\","
                     "\"update_size\":%" PRIu32 ","
                     "\"free_heap\":%" PRIu32
                     "}\n",
                     running ? running->label : "(unknown)",
                     boot    ? boot->label    : "(unknown)",
                     update  ? update->label  : "(unknown)",
                     update  ? update->size   : 0u,
                     esp_get_free_heap_size());
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, resp, n);
}

static esp_err_t ota_post_handler(httpd_req_t *req)
{
    const esp_partition_t *update = esp_ota_get_next_update_partition(NULL);
    if (!update) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "no OTA partition available");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "OTA target: %s @ 0x%08" PRIx32 " size=%" PRIu32,
             update->label, update->address, update->size);
    ESP_LOGI(TAG, "incoming payload: %d bytes", req->content_len);

    esp_ota_handle_t handle = 0;
    esp_err_t err = esp_ota_begin(update, OTA_WITH_SEQUENTIAL_WRITES, &handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_begin failed: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            esp_err_to_name(err));
        return ESP_FAIL;
    }

    char *buf = malloc(OTA_RECV_CHUNK);
    if (!buf) {
        esp_ota_abort(handle);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "oom");
        return ESP_FAIL;
    }

    int total = 0;
    int remaining = req->content_len;
    while (remaining > 0) {
        int want = remaining < OTA_RECV_CHUNK ? remaining : OTA_RECV_CHUNK;
        int r = httpd_req_recv(req, buf, want);
        if (r == HTTPD_SOCK_ERR_TIMEOUT) continue;
        if (r <= 0) {
            ESP_LOGE(TAG, "recv failed at %d/%d (r=%d)",
                     total, req->content_len, r);
            free(buf);
            esp_ota_abort(handle);
            return ESP_FAIL;
        }
        err = esp_ota_write(handle, buf, r);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "esp_ota_write failed at %d: %s",
                     total, esp_err_to_name(err));
            free(buf);
            esp_ota_abort(handle);
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                                esp_err_to_name(err));
            return ESP_FAIL;
        }
        total += r;
        remaining -= r;
    }
    free(buf);

    err = esp_ota_end(handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_end failed: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            esp_err_to_name(err));
        return ESP_FAIL;
    }

    err = esp_ota_set_boot_partition(update);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "set_boot_partition failed: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            esp_err_to_name(err));
        return ESP_FAIL;
    }

    char resp[192];
    int n = snprintf(resp, sizeof resp,
                     "{\"ok\":true,\"bytes\":%d,\"next_partition\":\"%s\","
                     "\"rebooting_in_ms\":800}\n",
                     total, update->label);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, resp, n);

    ESP_LOGI(TAG, "OTA complete (%d bytes → %s), rebooting",
             total, update->label);
    vTaskDelay(pdMS_TO_TICKS(800));
    esp_restart();
    return ESP_OK;  /* unreachable */
}

/* -------------------------------------------------------------------------- */
/* HTTP server                                                                */
/* -------------------------------------------------------------------------- */

static httpd_handle_t http_start(void)
{
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.server_port = CONFIG_TEST_OTA_HTTP_PORT;
    cfg.max_uri_handlers = 8;
    cfg.recv_wait_timeout = 30;
    cfg.send_wait_timeout = 30;

    httpd_handle_t server = NULL;
    esp_err_t err = httpd_start(&server, &cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "httpd_start failed: %s", esp_err_to_name(err));
        return NULL;
    }

    const httpd_uri_t root = {
        .uri = "/", .method = HTTP_GET, .handler = root_handler,
    };
    const httpd_uri_t info = {
        .uri = "/info", .method = HTTP_GET, .handler = info_handler,
    };
    const httpd_uri_t ota = {
        .uri = "/ota", .method = HTTP_POST, .handler = ota_post_handler,
    };
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &root));
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &info));
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &ota));

    ESP_LOGI(TAG, "HTTP server up on :%d — endpoints: /  /info  /ota",
             CONFIG_TEST_OTA_HTTP_PORT);
    return server;
}

/* -------------------------------------------------------------------------- */
/* app_main                                                                   */
/* -------------------------------------------------------------------------- */

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    const esp_partition_t *running = esp_ota_get_running_partition();
    ESP_LOGI(TAG, "boot ok — running from %s @ 0x%08" PRIx32,
             running ? running->label : "(unknown)",
             running ? running->address : 0u);

    wifi_init();

    /* Esperá hasta 20s a que llegue la IP; si no, seguimos igual y el server
     * arranca — útil para diagnosticar con monitor serie. */
    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_evt, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, pdMS_TO_TICKS(20000));
    if ((bits & WIFI_CONNECTED_BIT) == 0) {
        ESP_LOGW(TAG, "no IP after 20s — starting server anyway; retry WiFi in bg");
    }

    http_start();

    int i = 0;
    while (1) {
        ESP_LOGI(TAG, "heartbeat %d — free_heap=%" PRIu32 "B",
                 i++, esp_get_free_heap_size());
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

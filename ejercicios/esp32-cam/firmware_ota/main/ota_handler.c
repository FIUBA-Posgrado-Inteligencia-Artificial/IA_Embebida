#include "ota_handler.h"
#include "sdkconfig.h"

#ifdef CONFIG_ESP32CAM_OTA_ENABLED

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "ota";

#define OTA_RECV_CHUNK 4096

static esp_err_t ota_post_handler(httpd_req_t *req)
{
    const esp_partition_t *update = esp_ota_get_next_update_partition(NULL);
    if (!update) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "no OTA partition available");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "target partition: %s @ 0x%08" PRIx32 " size=%" PRIu32,
             update->label, update->address, update->size);
    ESP_LOGI(TAG, "incoming OTA payload: %d bytes", req->content_len);

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

    char resp[160];
    int n = snprintf(resp, sizeof resp,
                     "{\"ok\":true,\"bytes\":%d,\"next_partition\":\"%s\","
                     "\"rebooting_in_ms\":800}\n",
                     total, update->label);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, resp, n);

    ESP_LOGI(TAG, "OTA complete (%d bytes written to %s), rebooting",
             total, update->label);
    vTaskDelay(pdMS_TO_TICKS(800));
    esp_restart();
    return ESP_OK;  /* unreachable */
}

static esp_err_t ota_info_handler(httpd_req_t *req)
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
                     "\"update_size\":%" PRIu32
                     "}\n",
                     running ? running->label : "(unknown)",
                     boot    ? boot->label    : "(unknown)",
                     update  ? update->label  : "(unknown)",
                     update  ? update->size   : 0u);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, resp, n);
}

esp_err_t ota_handler_register(httpd_handle_t server)
{
    if (!server) {
        ESP_LOGW(TAG, "no httpd server — OTA endpoint not registered");
        return ESP_ERR_INVALID_STATE;
    }
    const httpd_uri_t post = {
        .uri = "/ota", .method = HTTP_POST, .handler = ota_post_handler,
    };
    const httpd_uri_t info = {
        .uri = "/ota/info", .method = HTTP_GET, .handler = ota_info_handler,
    };
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &post));
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &info));
    ESP_LOGI(TAG, "OTA endpoints ready: POST /ota  GET /ota/info");
    return ESP_OK;
}

#else

esp_err_t ota_handler_register(httpd_handle_t server)
{
    (void)server;
    return ESP_OK;
}

#endif

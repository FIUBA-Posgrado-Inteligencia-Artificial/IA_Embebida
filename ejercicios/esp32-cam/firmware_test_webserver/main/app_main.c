#include <stdio.h>
#include <string.h>

#include "esp_camera.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "wifi_creds.h"

static const char *TAG = "test_webserver";

/* AI-Thinker ESP32-CAM pin map. */
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK     0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27
#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      21
#define CAM_PIN_D2      19
#define CAM_PIN_D1      18
#define CAM_PIN_D0       5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

static const camera_config_t CAM_CFG = {
    .pin_pwdn       = CAM_PIN_PWDN,
    .pin_reset      = CAM_PIN_RESET,
    .pin_xclk       = CAM_PIN_XCLK,
    .pin_sccb_sda   = CAM_PIN_SIOD,
    .pin_sccb_scl   = CAM_PIN_SIOC,
    .pin_d7         = CAM_PIN_D7,
    .pin_d6         = CAM_PIN_D6,
    .pin_d5         = CAM_PIN_D5,
    .pin_d4         = CAM_PIN_D4,
    .pin_d3         = CAM_PIN_D3,
    .pin_d2         = CAM_PIN_D2,
    .pin_d1         = CAM_PIN_D1,
    .pin_d0         = CAM_PIN_D0,
    .pin_vsync      = CAM_PIN_VSYNC,
    .pin_href       = CAM_PIN_HREF,
    .pin_pclk       = CAM_PIN_PCLK,

    .xclk_freq_hz   = 20000000,
    .ledc_timer     = LEDC_TIMER_0,
    .ledc_channel   = LEDC_CHANNEL_0,

    .pixel_format   = PIXFORMAT_JPEG,
    .frame_size     = FRAMESIZE_VGA,
    .jpeg_quality   = 12,
    .fb_count       = 2,
    .fb_location    = CAMERA_FB_IN_PSRAM,
    .grab_mode      = CAMERA_GRAB_WHEN_EMPTY,
};

#define STREAM_BOUNDARY "ESPFRAME"
#define STREAM_PART_HEADER \
    "--" STREAM_BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n"

static const char INDEX_HTML[] =
    "<!DOCTYPE html><html><head>"
    "<meta charset='utf-8'>"
    "<title>ESP32-CAM test webserver</title>"
    "<style>"
    "body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0;padding:1em}"
    "img{max-width:100%;border:2px solid #444;border-radius:4px}"
    "h3{font-size:1.1em;margin:.5em 0}"
    "</style></head><body>"
    "<h3>ESP32-CAM — test webserver</h3>"
    "<img src='/stream'>"
    "</body></html>";

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "disconnected, retrying in 2s");
        vTaskDelay(pdMS_TO_TICKS(2000));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "connected, IP=" IPSTR, IP2STR(&e->ip_info.ip));
    }
}

static void wifi_init_sta(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t icfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&icfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL, NULL));

    wifi_config_t cfg = { 0 };
    strncpy((char *)cfg.sta.ssid,     WIFI_SSID,     sizeof(cfg.sta.ssid));
    strncpy((char *)cfg.sta.password, WIFI_PASSWORD, sizeof(cfg.sta.password));
    cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "connecting to \"%s\"", WIFI_SSID);
}

static esp_err_t camera_init(void)
{
    esp_err_t err = esp_camera_init(&CAM_CFG);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed: %s", esp_err_to_name(err));
        return err;
    }
    ESP_LOGI(TAG, "camera init OK — VGA JPEG");
    return ESP_OK;
}

static esp_err_t index_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, INDEX_HTML, sizeof(INDEX_HTML) - 1);
}

static esp_err_t stream_handler(httpd_req_t *req)
{
    esp_err_t res = httpd_resp_set_type(req,
        "multipart/x-mixed-replace;boundary=" STREAM_BOUNDARY);
    if (res != ESP_OK) return res;

    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    httpd_resp_set_hdr(req, "Pragma", "no-cache");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    char part_hdr[96];
    ESP_LOGI(TAG, "stream client connected");

    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb == NULL) {
            ESP_LOGW(TAG, "frame capture failed");
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }

        int n = snprintf(part_hdr, sizeof part_hdr, STREAM_PART_HEADER, fb->len);
        res = httpd_resp_send_chunk(req, part_hdr, n);
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, "\r\n", 2);

        esp_camera_fb_return(fb);

        if (res != ESP_OK) break;  /* client disconnected */
    }

    ESP_LOGI(TAG, "stream client disconnected");
    return res;
}

static esp_err_t start_httpd(void)
{
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.server_port       = 80;
    cfg.ctrl_port         = 81;      /* must differ from server_port */
    cfg.recv_wait_timeout = 10;
    cfg.send_wait_timeout = 10;
    cfg.lru_purge_enable  = true;

    httpd_handle_t server = NULL;
    esp_err_t err = httpd_start(&server, &cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "httpd_start failed: %s", esp_err_to_name(err));
        return err;
    }

    const httpd_uri_t index_uri  = {
        .uri = "/", .method = HTTP_GET, .handler = index_handler };
    const httpd_uri_t stream_uri = {
        .uri = "/stream", .method = HTTP_GET, .handler = stream_handler };
    httpd_register_uri_handler(server, &index_uri);
    httpd_register_uri_handler(server, &stream_uri);

    ESP_LOGI(TAG, "HTTP server started on port 80 — open http://<IP>/");
    return ESP_OK;
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    if (camera_init() != ESP_OK) {
        ESP_LOGE(TAG, "rebooting in 3s");
        vTaskDelay(pdMS_TO_TICKS(3000));
        esp_restart();
    }

    wifi_init_sta();
    start_httpd();

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

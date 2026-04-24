#include "http_stream_server.h"
#include "sdkconfig.h"

#ifdef CONFIG_ESP32CAM_HTTP_STREAM_ENABLED

#include "camera.h"
#include "esp_log.h"
#include "esp_http_server.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "http_stream";

#define BOUNDARY    "ESPFRAME"
#define PART_HEADER "--" BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n"

static const char INDEX_HTML[] =
    "<!DOCTYPE html><html><head>"
    "<meta charset='utf-8'>"
    "<title>ESP32-CAM</title>"
    "<style>"
    "body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0;padding:1em}"
    "img{max-width:100%;border:2px solid #444;border-radius:4px}"
    "h3{font-size:1.1em;margin:.5em 0}"
    "</style></head><body>"
    "<h3>ESP32-CAM — live stream</h3>"
    "<img src='/stream'>"
    "</body></html>";

static esp_err_t index_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, INDEX_HTML, sizeof(INDEX_HTML) - 1);
}

static esp_err_t stream_handler(httpd_req_t *req)
{
    esp_err_t res = httpd_resp_set_type(req,
        "multipart/x-mixed-replace;boundary=" BOUNDARY);
    if (res != ESP_OK) return res;

    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    httpd_resp_set_hdr(req, "Pragma", "no-cache");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    char part_hdr[96];
    ESP_LOGI(TAG, "stream client connected");

    while (true) {
        camera_fb_t *fb = NULL;
        if (camera_capture_jpeg(&fb) != ESP_OK || fb == NULL) {
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }

        int n = snprintf(part_hdr, sizeof part_hdr, PART_HEADER, fb->len);
        res = httpd_resp_send_chunk(req, part_hdr, n);
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, "\r\n", 2);

        camera_return(fb);

        if (res != ESP_OK) break;  /* cliente desconectado */
    }

    ESP_LOGI(TAG, "stream client disconnected");
    return res;
}

esp_err_t http_stream_server_start(void)
{
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.server_port       = CONFIG_ESP32CAM_HTTP_STREAM_PORT;
    /* ctrl_port debe ser distinto de server_port */
    cfg.ctrl_port         = (uint16_t)(CONFIG_ESP32CAM_HTTP_STREAM_PORT + 1);
    cfg.recv_wait_timeout = 10;
    cfg.send_wait_timeout = 10;
    cfg.lru_purge_enable  = true;

    httpd_handle_t server = NULL;
    esp_err_t err = httpd_start(&server, &cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "start failed: %s", esp_err_to_name(err));
        return err;
    }

    const httpd_uri_t index_uri  = { .uri = "/",       .method = HTTP_GET, .handler = index_handler };
    const httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler };
    httpd_register_uri_handler(server, &index_uri);
    httpd_register_uri_handler(server, &stream_uri);

    ESP_LOGI(TAG, "HTTP server started on port %d  →  http://<IP>/",
             CONFIG_ESP32CAM_HTTP_STREAM_PORT);
    return ESP_OK;
}

#else

esp_err_t http_stream_server_start(void) { return ESP_OK; }

#endif

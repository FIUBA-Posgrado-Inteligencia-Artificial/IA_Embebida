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

    /* XCLK 10 MHz (standard is 20 MHz). Halving the sensor master clock
     * gives the OV2640's internal state machine more time per tick — SCCB
     * writes during init are the most timing-sensitive step and were
     * failing consistently at 20 MHz on this board. Trade-off: max frame
     * rate scales with XCLK, but for VGA MJPEG we're bandwidth-limited
     * by WiFi anyway, not by the sensor. */
    .xclk_freq_hz   = 10000000,
    .ledc_timer     = LEDC_TIMER_0,
    .ledc_channel   = LEDC_CHANNEL_0,

    .pixel_format   = PIXFORMAT_JPEG,
    /* SVGA (800×600) over VGA (640×480): the esp32-camera driver sizes the
     * JPEG frame buffer as (width*height/5), so SVGA gives 96000 bytes vs
     * VGA's 61440. The extra headroom eliminates the occasional buffer
     * overruns that produced NO-EOI errors at VGA. Subjectively also a
     * sharper-looking feed. */
    .frame_size     = FRAMESIZE_SVGA,
    /* jpeg_quality: 0-63, lower = better quality = larger file. 12 is a
     * solid middle ground at SVGA. */
    .jpeg_quality   = 12,
    /* fb_count=1: single frame buffer, driver can't return a half-filled
     * one. fb_count=2 with an aggressive MJPEG consumer triggered races in
     * cam_hal that produced NO-SOI floods. Frame rate drops slightly but
     * we're network-limited over WiFi anyway. */
    .fb_count       = 1,
    .fb_location    = CAMERA_FB_IN_PSRAM,
    /* GRAB_LATEST is still correct: driver returns the most recent
     * complete frame; with fb_count=1 this means it blocks until DMA
     * finishes the current frame. */
    .grab_mode      = CAMERA_GRAB_LATEST,
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
    /* Con password vacío la red es abierta: si dejamos el threshold en WPA2_PSK
     * el driver descarta el AP y nunca asocia. */
    cfg.sta.threshold.authmode =
        (WIFI_PASSWORD[0] == '\0') ? WIFI_AUTH_OPEN : WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "connecting to \"%s\"", WIFI_SSID);
}

/* Camera init retry parameters — ported from firmware/main/camera.c.
 * OV2640 SCCB writes fail intermittently on AI-Thinker cold boot; the
 * sensor needs long idle between the driver's own PWDN toggle and a
 * retry. 500 ms initial settle, 1 s between attempts, max 5 attempts. */
#define CAM_INIT_SETTLE_MS     500
#define CAM_INIT_RETRY_MS     1000
#define CAM_INIT_MAX_ATTEMPTS    5

static esp_err_t camera_init(void)
{
    /* Let OV2640 VCC settle before any SCCB traffic. */
    vTaskDelay(pdMS_TO_TICKS(CAM_INIT_SETTLE_MS));

    for (int attempt = 1; attempt <= CAM_INIT_MAX_ATTEMPTS; attempt++) {
        esp_err_t err = esp_camera_init(&CAM_CFG);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "camera init OK — VGA JPEG (attempt %d)", attempt);
            return ESP_OK;
        }
        ESP_LOGW(TAG, "camera init failed (attempt %d): %s",
                 attempt, esp_err_to_name(err));
        /* Release GPIO/DMA/ISR state so the next esp_camera_init starts
         * from a clean slate — without this, retries skip fresh PWDN and
         * tend to fail again on the same register write. */
        esp_camera_deinit();
        vTaskDelay(pdMS_TO_TICKS(CAM_INIT_RETRY_MS));
    }
    ESP_LOGE(TAG, "camera init failed after %d attempts", CAM_INIT_MAX_ATTEMPTS);
    return ESP_FAIL;
}

static esp_err_t index_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, INDEX_HTML, sizeof(INDEX_HTML) - 1);
}

/* Log first 8 and last 2 bytes of a JPEG frame. A valid JPEG starts with
 * FF D8 FF E0 (SOI + APP0) and ends with FF D9 (EOI). Anything else
 * means cam_hal delivered a malformed buffer. */
static void log_jpeg_bytes(const char *tag, const uint8_t *buf, size_t len)
{
    if (len < 10) {
        ESP_LOGW(TAG, "%s: frame too short (%u bytes)", tag, (unsigned)len);
        return;
    }
    ESP_LOGI(TAG,
        "%s: len=%u  SOI=%02X %02X %02X %02X %02X %02X %02X %02X  EOI=%02X %02X",
        tag, (unsigned)len,
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        buf[len - 2], buf[len - 1]);
}

/* Single-frame snapshot endpoint — serves one JPEG as a plain image/jpeg
 * response, no multipart. Good for isolating "is individual frame capture
 * working?" from "is the MJPEG stream loop working?". Open /snap in the
 * browser to see. */
static esp_err_t snap_handler(httpd_req_t *req)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb == NULL) {
        ESP_LOGE(TAG, "snap: fb_get returned NULL");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    log_jpeg_bytes("snap", fb->buf, fb->len);

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=snap.jpg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return res;
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

    int null_retries = 0;
    unsigned frame_count = 0;
    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb == NULL) {
            if (++null_retries >= 10) {
                ESP_LOGE(TAG, "camera unresponsive, closing stream");
                return ESP_FAIL;
            }
            ESP_LOGW(TAG, "frame capture failed (%d/10)", null_retries);
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }
        null_retries = 0;

        /* Debug: log the first frame plus every 30th after that, so we
         * can see what we're actually sending without flooding the log. */
        if (frame_count == 0 || frame_count % 30 == 0) {
            char tag[32];
            snprintf(tag, sizeof tag, "stream[%u]", frame_count);
            log_jpeg_bytes(tag, fb->buf, fb->len);
        }
        frame_count++;

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
    const httpd_uri_t snap_uri = {
        .uri = "/snap", .method = HTTP_GET, .handler = snap_handler };
    httpd_register_uri_handler(server, &index_uri);
    httpd_register_uri_handler(server, &stream_uri);
    httpd_register_uri_handler(server, &snap_uri);

    ESP_LOGI(TAG, "HTTP server started on port 80 — /  /stream  /snap");
    return ESP_OK;
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    /* WiFi FIRST, then camera. Rationale: with this AI-Thinker board's
     * marginal 3.3V supply, activating the PHY peripheral while the
     * camera is already consuming power dips the rail hard enough to
     * trip a brownout interrupt (shows as rst:0x3 SW_RESET in ESP-IDF
     * 5.x, not the classic RTCWDT_BROWN_OUT_RESET, because
     * CONFIG_ESP_SYSTEM_BROWNOUT_INTR=y). Initializing WiFi first means
     * phy_init runs with only PSRAM active, so the rail stays above
     * the 2.43V threshold. Once WiFi is stable, camera_init can run —
     * camera's steady-state current draw is well within budget. */
    wifi_init_sta();

    if (camera_init() != ESP_OK) {
        ESP_LOGE(TAG, "rebooting in 3s");
        vTaskDelay(pdMS_TO_TICKS(3000));
        esp_restart();
    }

    if (start_httpd() != ESP_OK) {
        ESP_LOGE(TAG, "httpd start failed, rebooting in 3s");
        vTaskDelay(pdMS_TO_TICKS(3000));
        esp_restart();
    }

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

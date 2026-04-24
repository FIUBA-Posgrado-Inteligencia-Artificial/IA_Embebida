#include "camera.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

static const char *TAG = "camera";

/* Mutex that serializes camera access between inference (RGB565) and HTTP
 * stream (JPEG). Acquired by camera_capture*(), released by camera_return(). */
static SemaphoreHandle_t g_cam_mutex = NULL;

/* Current sensor pixel format. Track it so we only call set_pixformat when we
 * actually need to switch (it drops the first post-switch frame). */
static pixformat_t g_current_fmt = PIXFORMAT_RGB565;

/* Cold-boot settle before first SCCB traffic. OV2640 needs time after VCC
 * ramps before its register state machine is ready for writes. */
#define CAM_INIT_SETTLE_MS 500

/* Retry spacing. 1s is what empirically works — the sensor needs long idle
 * between power-cycles from the driver's own PWDN toggle. Shorter delays
 * catch the sensor still half-initialized and SCCB writes time out. */
#define CAM_INIT_RETRY_MS 1000

#define CAM_INIT_MAX_ATTEMPTS 5

/* Pines AI-Thinker ESP32-CAM */
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

    .pixel_format   = PIXFORMAT_RGB565,
    .frame_size     = FRAMESIZE_QVGA,   /* 320x240 — native OV2640 size; fits in PSRAM */
    .jpeg_quality   = 0,
    .fb_count       = 1,
    .fb_location    = CAMERA_FB_IN_PSRAM,
    .grab_mode      = CAMERA_GRAB_WHEN_EMPTY,
};

esp_err_t camera_init(void)
{
    /* Let OV2640 settle after power-on before any SCCB traffic. */
    vTaskDelay(pdMS_TO_TICKS(CAM_INIT_SETTLE_MS));

    for (int attempt = 1; attempt <= CAM_INIT_MAX_ATTEMPTS; attempt++) {
        esp_err_t err = esp_camera_init(&CAM_CFG);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "OV2640 init OK (attempt %d)", attempt);
            g_cam_mutex = xSemaphoreCreateMutex();
            g_current_fmt = CAM_CFG.pixel_format;
            return g_cam_mutex != NULL ? ESP_OK : ESP_ERR_NO_MEM;
        }
        ESP_LOGW(TAG, "init failed (attempt %d): %s", attempt, esp_err_to_name(err));
        /* Release any GPIO/DMA/ISR state the failed init left behind so the
         * next esp_camera_init starts from a clean slate. Without this, the
         * re-init skips fresh PWDN toggle + ISR install and tends to fail
         * again on the same sensor config write. */
        esp_camera_deinit();
        vTaskDelay(pdMS_TO_TICKS(CAM_INIT_RETRY_MS));
    }
    ESP_LOGE(TAG, "init failed after %d attempts", CAM_INIT_MAX_ATTEMPTS);
    return ESP_FAIL;
}

/* Caller must hold g_cam_mutex. */
static esp_err_t switch_format_locked(pixformat_t fmt)
{
    if (fmt == g_current_fmt) return ESP_OK;
    sensor_t *s = esp_camera_sensor_get();
    if (!s || s->set_pixformat(s, fmt) != 0) {
        ESP_LOGE(TAG, "set_pixformat(%d) failed", fmt);
        return ESP_FAIL;
    }
    g_current_fmt = fmt;
    /* Drop one stale frame captured with the old format. */
    camera_fb_t *stale = esp_camera_fb_get();
    if (stale) esp_camera_fb_return(stale);
    return ESP_OK;
}

static esp_err_t capture_with_format(pixformat_t fmt, camera_fb_t **fb)
{
    if (!g_cam_mutex) return ESP_ERR_INVALID_STATE;
    if (xSemaphoreTake(g_cam_mutex, portMAX_DELAY) != pdTRUE) return ESP_FAIL;

    esp_err_t err = switch_format_locked(fmt);
    if (err != ESP_OK) {
        xSemaphoreGive(g_cam_mutex);
        return err;
    }

    *fb = esp_camera_fb_get();
    if (*fb == NULL) {
        ESP_LOGW(TAG, "frame capture failed");
        xSemaphoreGive(g_cam_mutex);
        return ESP_FAIL;
    }
    /* mutex stays held until camera_return() */
    return ESP_OK;
}

esp_err_t camera_capture(camera_fb_t **fb)
{
    return capture_with_format(PIXFORMAT_RGB565, fb);
}

esp_err_t camera_capture_jpeg(camera_fb_t **fb)
{
    return capture_with_format(PIXFORMAT_JPEG, fb);
}

void camera_return(camera_fb_t *fb)
{
    if (fb) esp_camera_fb_return(fb);
    if (g_cam_mutex) xSemaphoreGive(g_cam_mutex);
}

#include "camera.h"
#include "board.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#if BOARD_HAS_CAMERA

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

static const camera_config_t CAM_CFG = {
    .pin_pwdn       = BOARD_CAM_PIN_PWDN,
    .pin_reset      = BOARD_CAM_PIN_RESET,
    .pin_xclk       = BOARD_CAM_PIN_XCLK,
    .pin_sccb_sda   = BOARD_CAM_PIN_SIOD,
    .pin_sccb_scl   = BOARD_CAM_PIN_SIOC,
    .pin_d7         = BOARD_CAM_PIN_D7,
    .pin_d6         = BOARD_CAM_PIN_D6,
    .pin_d5         = BOARD_CAM_PIN_D5,
    .pin_d4         = BOARD_CAM_PIN_D4,
    .pin_d3         = BOARD_CAM_PIN_D3,
    .pin_d2         = BOARD_CAM_PIN_D2,
    .pin_d1         = BOARD_CAM_PIN_D1,
    .pin_d0         = BOARD_CAM_PIN_D0,
    .pin_vsync      = BOARD_CAM_PIN_VSYNC,
    .pin_href       = BOARD_CAM_PIN_HREF,
    .pin_pclk       = BOARD_CAM_PIN_PCLK,

    .xclk_freq_hz   = BOARD_CAM_XCLK_FREQ_HZ,
    .ledc_timer     = LEDC_TIMER_0,
    .ledc_channel   = LEDC_CHANNEL_0,

    .pixel_format   = PIXFORMAT_RGB565,
    /* Sin PSRAM el framebuffer vive en DRAM: QVGA RGB565 son 150 KB y no
     * dejan lugar para los buffers de WiFi (rx init falla con 9/10 y el
     * boot entra en loop). QQVGA (160x120, 38 KB) deja ~110 KB libres y
     * el preprocess resizea igual desde cualquier resolución. */
    .frame_size     = EFFECTIVE_PSRAM ? FRAMESIZE_QVGA : FRAMESIZE_QQVGA,
    .jpeg_quality   = 0,
    .fb_count       = 1,
    .fb_location    = EFFECTIVE_PSRAM ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM,
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

    /* Con fb_count=1 + GRAB_WHEN_EMPTY, el buffer ya contiene un frame
     * capturado apenas se liberó el anterior — la escena de la captura
     * PASADA. Lo descartamos para que el frame devuelto empiece a
     * exponerse recién ahora (cuesta ~1 frame de latencia, ~60 ms). */
    camera_fb_t *stale = esp_camera_fb_get();
    if (stale) esp_camera_fb_return(stale);

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

#else /* !BOARD_HAS_CAMERA */

esp_err_t camera_init(void) { return ESP_FAIL; }
esp_err_t camera_capture(camera_fb_t **fb) { (void)fb; return ESP_FAIL; }
esp_err_t camera_capture_jpeg(camera_fb_t **fb) { (void)fb; return ESP_FAIL; }
void camera_return(camera_fb_t *fb) { (void)fb; }

#endif /* BOARD_HAS_CAMERA */

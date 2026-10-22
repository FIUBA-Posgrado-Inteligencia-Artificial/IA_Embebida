#include "capture_state.h"
#include "board.h"
#include <string.h>
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static SemaphoreHandle_t s_mutex;
static capture_status_t  s_status;   /* .mode = SINGLE, .seq = 0 */
static uint8_t          *s_jpeg;     /* PSRAM, realloc por captura */
static size_t            s_jpeg_len;
static size_t            s_jpeg_cap;

esp_err_t capture_state_init(void)
{
    s_mutex = xSemaphoreCreateMutex();
    if (s_mutex == NULL) return ESP_ERR_NO_MEM;
#if START_IN_CONTINUOUS
    s_status.mode = CAPTURE_MODE_CONTINUOUS;
#endif
    return ESP_OK;
}

capture_mode_t capture_state_get_mode(void)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    capture_mode_t m = s_status.mode;
    xSemaphoreGive(s_mutex);
    return m;
}

capture_mode_t capture_state_set_mode(capture_mode_t m)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    s_status.mode = m;
    xSemaphoreGive(s_mutex);
    return m;
}

capture_mode_t capture_state_toggle_mode(void)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    s_status.mode = (s_status.mode == CAPTURE_MODE_SINGLE)
                    ? CAPTURE_MODE_CONTINUOUS : CAPTURE_MODE_SINGLE;
    capture_mode_t m = s_status.mode;
    xSemaphoreGive(s_mutex);
    return m;
}

void capture_state_get_status(capture_status_t *out)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    *out = s_status;
    xSemaphoreGive(s_mutex);
}

uint32_t capture_state_publish_result(const inference_result_t *r,
                                      const uint8_t *jpg, size_t jpg_len,
                                      uint32_t t_cap_ms, uint32_t t_pre_ms,
                                      uint32_t t_inf_ms, float fps)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    s_status.seq++;
    s_status.result          = *r;
    s_status.t_capture_ms    = t_cap_ms;
    s_status.t_preprocess_ms = t_pre_ms;
    s_status.t_inference_ms  = t_inf_ms;
    s_status.fps             = fps;
    s_status.last_error[0]   = '\0';

    if (jpg != NULL && jpg_len > 0) {
        if (jpg_len > s_jpeg_cap) {
            uint8_t *nb = heap_caps_realloc(s_jpeg, jpg_len,
                              EFFECTIVE_PSRAM ? MALLOC_CAP_SPIRAM : MALLOC_CAP_8BIT);
            if (nb != NULL) {
                s_jpeg = nb;
                s_jpeg_cap = jpg_len;
            }
        }
        if (jpg_len <= s_jpeg_cap) {
            memcpy(s_jpeg, jpg, jpg_len);
            s_jpeg_len = jpg_len;
        } else {
            strlcpy(s_status.last_error, "jpeg alloc failed (imagen vieja)",
                    sizeof s_status.last_error);
        }
    } else {
        strlcpy(s_status.last_error, "jpeg encode failed (imagen vieja)",
                sizeof s_status.last_error);
    }

    uint32_t seq = s_status.seq;
    xSemaphoreGive(s_mutex);
    return seq;
}

void capture_state_publish_error(const char *msg)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    strlcpy(s_status.last_error, msg, sizeof s_status.last_error);
    xSemaphoreGive(s_mutex);
}

void capture_state_borrow_jpeg(const uint8_t **buf, size_t *len)
{
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    *buf = s_jpeg_len ? s_jpeg : NULL;
    *len = s_jpeg_len;
    /* mutex queda tomado hasta release — el caller manda el buffer y libera */
}

void capture_state_release_jpeg(void)
{
    xSemaphoreGive(s_mutex);
}

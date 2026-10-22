#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"
#include "inference.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { CAPTURE_MODE_SINGLE = 0, CAPTURE_MODE_CONTINUOUS = 1 } capture_mode_t;

typedef struct {
    uint32_t           seq;              /* 0 = aún no hubo captura */
    capture_mode_t     mode;
    inference_result_t result;
    uint32_t           t_capture_ms;
    uint32_t           t_preprocess_ms;
    uint32_t           t_inference_ms;
    float              fps;              /* 0 en modo single */
    char               last_error[64];   /* "" si ok */
} capture_status_t;

esp_err_t capture_state_init(void);
capture_mode_t capture_state_get_mode(void);
/* devuelve el modo resultante */
capture_mode_t capture_state_set_mode(capture_mode_t m);
capture_mode_t capture_state_toggle_mode(void);
/* copia el status completo (thread-safe) */
void capture_state_get_status(capture_status_t *out);
/* Publica atómicamente resultado + JPEG (copia en PSRAM) bajo el mismo lock,
 * incrementando seq una sola vez — /result y /last.jpg siempre corresponden
 * a la misma captura. `jpg` puede ser NULL (encode falló): se conserva la
 * imagen anterior y se deja last_error explicándolo. Devuelve el seq nuevo. */
uint32_t capture_state_publish_result(const inference_result_t *r,
                                      const uint8_t *jpg, size_t jpg_len,
                                      uint32_t t_cap_ms, uint32_t t_pre_ms,
                                      uint32_t t_inf_ms, float fps);
void capture_state_publish_error(const char *msg);
/* lockea y presta el jpeg; devolver con capture_state_release_jpeg().
 * *buf=NULL, *len=0 si no hay. */
void capture_state_borrow_jpeg(const uint8_t **buf, size_t *len);
void capture_state_release_jpeg(void);

#ifdef __cplusplus
}
#endif

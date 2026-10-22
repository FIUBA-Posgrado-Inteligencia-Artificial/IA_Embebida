#pragma once
#include "esp_err.h"
#include "esp_camera.h"

/* Inicializa la OV2640 con pines AI-Thinker en modo RGB565 por defecto.
 * Reintenta hasta 5 veces con 1s de espera (más 500ms de settle inicial). */
esp_err_t camera_init(void);

/* Captura en RGB565 (para inferencia). Toma mutex exclusivo — el caller debe
 * llamar a camera_return() para liberar el frame + mutex. Hace pixformat switch
 * si la última captura fue JPEG. */
esp_err_t camera_capture(camera_fb_t **fb);

/* Captura en JPEG (para stream HTTP). Misma semántica de mutex que camera_capture. */
esp_err_t camera_capture_jpeg(camera_fb_t **fb);

/* Devuelve el frame al driver (libera buffer interno) y el mutex. */
void camera_return(camera_fb_t *fb);

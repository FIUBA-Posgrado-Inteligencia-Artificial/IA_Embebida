#pragma once
#include <stdint.h>
#include <stdbool.h>

/* Pipeline completo: RGB565 src_w×src_h → int8 dst_w*dst_h*dst_channels.
 * mean/std vienen de model_data.h en escala uint8 [0,255].
 * Si use_bilinear==false, usa nearest neighbor. */
void preprocess_rgb565(const uint16_t *src,
                       int src_w, int src_h,
                       int8_t *dst,
                       int dst_w, int dst_h,
                       int dst_channels,
                       const uint8_t *mean,
                       const uint8_t *std,
                       bool use_bilinear);

/* ---- Helpers expuestos para tests de host (no para uso externo) ---- */

/* Convierte un pixel RGB565 (LE) a componentes RGB888. */
void preprocess_px_rgb565_to_rgb888(uint16_t px,
                                    uint8_t *r, uint8_t *g, uint8_t *b);

/* Muestra RGB888 del buffer RGB565 src en coords fraccionarias (fx, fy).
 * fx, fy en [0, src_w-1] y [0, src_h-1]. */
void preprocess_sample_nearest(const uint16_t *src, int src_w, int src_h,
                               float fx, float fy,
                               uint8_t *r, uint8_t *g, uint8_t *b);
void preprocess_sample_bilinear(const uint16_t *src, int src_w, int src_h,
                                float fx, float fy,
                                uint8_t *r, uint8_t *g, uint8_t *b);

/* Luminancia entera (Rec. 601): gray = (299R + 587G + 114B + 500)/1000. */
uint8_t preprocess_rgb_to_gray(uint8_t r, uint8_t g, uint8_t b);

/* Cuantización simétrica int8: x = (val-mean)*128/std, saturada a [-128,127]. */
int8_t preprocess_normalize_u8(uint8_t val, uint8_t mean, uint8_t std);

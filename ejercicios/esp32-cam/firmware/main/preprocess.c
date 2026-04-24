#include "preprocess.h"

void preprocess_px_rgb565_to_rgb888(uint16_t px,
                                    uint8_t *r, uint8_t *g, uint8_t *b)
{
    /* RGB565 layout: RRRRRGGG GGGBBBBB (MSB-first).
     * Escalar cada canal a 8 bits repitiendo los bits altos. */
    uint8_t r5 = (px >> 11) & 0x1F;
    uint8_t g6 = (px >>  5) & 0x3F;
    uint8_t b5 =  px        & 0x1F;
    *r = (uint8_t)((r5 << 3) | (r5 >> 2));
    *g = (uint8_t)((g6 << 2) | (g6 >> 4));
    *b = (uint8_t)((b5 << 3) | (b5 >> 2));
}

uint8_t preprocess_rgb_to_gray(uint8_t r, uint8_t g, uint8_t b)
{
    uint32_t sum = 299u*(uint32_t)r + 587u*(uint32_t)g + 114u*(uint32_t)b + 500u;
    return (uint8_t)(sum / 1000u);
}

int8_t preprocess_normalize_u8(uint8_t val, uint8_t mean, uint8_t std)
{
    if (std == 0) std = 1;  /* guard */
    int32_t x = ((int32_t)val - (int32_t)mean) * 128 / (int32_t)std;
    if (x >  127) x =  127;
    if (x < -128) x = -128;
    return (int8_t)x;
}

static int clamp_int(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

void preprocess_sample_nearest(const uint16_t *src, int src_w, int src_h,
                               float fx, float fy,
                               uint8_t *r, uint8_t *g, uint8_t *b)
{
    int x = (int)(fx + 0.5f);
    int y = (int)(fy + 0.5f);
    x = clamp_int(x, 0, src_w - 1);
    y = clamp_int(y, 0, src_h - 1);
    preprocess_px_rgb565_to_rgb888(src[y * src_w + x], r, g, b);
}

void preprocess_sample_bilinear(const uint16_t *src, int src_w, int src_h,
                                float fx, float fy,
                                uint8_t *r, uint8_t *g, uint8_t *b)
{
    if (fx < 0) fx = 0;
    if (fy < 0) fy = 0;
    if (fx > src_w - 1) fx = src_w - 1;
    if (fy > src_h - 1) fy = src_h - 1;

    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1 < src_w ? x0 + 1 : x0;
    int y1 = y0 + 1 < src_h ? y0 + 1 : y0;
    float ax = fx - (float)x0;
    float ay = fy - (float)y0;

    uint8_t r00,g00,b00,  r01,g01,b01,  r10,g10,b10,  r11,g11,b11;
    preprocess_px_rgb565_to_rgb888(src[y0*src_w+x0], &r00,&g00,&b00);
    preprocess_px_rgb565_to_rgb888(src[y0*src_w+x1], &r01,&g01,&b01);
    preprocess_px_rgb565_to_rgb888(src[y1*src_w+x0], &r10,&g10,&b10);
    preprocess_px_rgb565_to_rgb888(src[y1*src_w+x1], &r11,&g11,&b11);

    float top_r = (1-ax)*r00 + ax*r01;
    float top_g = (1-ax)*g00 + ax*g01;
    float top_b = (1-ax)*b00 + ax*b01;
    float bot_r = (1-ax)*r10 + ax*r11;
    float bot_g = (1-ax)*g10 + ax*g11;
    float bot_b = (1-ax)*b10 + ax*b11;

    *r = (uint8_t)((1-ay)*top_r + ay*bot_r + 0.5f);
    *g = (uint8_t)((1-ay)*top_g + ay*bot_g + 0.5f);
    *b = (uint8_t)((1-ay)*top_b + ay*bot_b + 0.5f);
}

void preprocess_rgb565(const uint16_t *src,
                       int src_w, int src_h,
                       int8_t *dst,
                       int dst_w, int dst_h,
                       int dst_channels,
                       const uint8_t *mean,
                       const uint8_t *std,
                       bool use_bilinear)
{
    const float sx = (float)src_w / (float)dst_w;
    const float sy = (float)src_h / (float)dst_h;

    for (int y = 0; y < dst_h; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        for (int x = 0; x < dst_w; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;

            uint8_t r, g, b;
            if (use_bilinear) {
                preprocess_sample_bilinear(src, src_w, src_h, fx, fy, &r, &g, &b);
            } else {
                preprocess_sample_nearest(src, src_w, src_h, fx, fy, &r, &g, &b);
            }

            if (dst_channels == 1) {
                uint8_t gray = preprocess_rgb_to_gray(r, g, b);
                dst[y*dst_w + x] = preprocess_normalize_u8(gray, mean[0], std[0]);
            } else {
                int idx = (y*dst_w + x) * 3;
                dst[idx + 0] = preprocess_normalize_u8(r, mean[0], std[0]);
                dst[idx + 1] = preprocess_normalize_u8(g, mean[1], std[1]);
                dst[idx + 2] = preprocess_normalize_u8(b, mean[2], std[2]);
            }
        }
    }
}

#include "unity.h"
#include "preprocess.h"

void setUp(void) {}
void tearDown(void) {}

static void fill_solid_rgb565(uint16_t *buf, int n, uint16_t px) {
    for (int i = 0; i < n; i++) buf[i] = px;
}

static void test_rgb565_black(void) {
    uint8_t r=1, g=1, b=1;
    preprocess_px_rgb565_to_rgb888(0x0000, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(0, r);
    TEST_ASSERT_EQUAL_UINT8(0, g);
    TEST_ASSERT_EQUAL_UINT8(0, b);
}

static void test_rgb565_white(void) {
    uint8_t r, g, b;
    preprocess_px_rgb565_to_rgb888(0xFFFF, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(255, r);
    TEST_ASSERT_EQUAL_UINT8(255, g);
    TEST_ASSERT_EQUAL_UINT8(255, b);
}

static void test_rgb565_pure_red(void) {
    // R=31 (5b), G=0, B=0 → 0xF800
    uint8_t r, g, b;
    preprocess_px_rgb565_to_rgb888(0xF800, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(255, r);
    TEST_ASSERT_EQUAL_UINT8(0,   g);
    TEST_ASSERT_EQUAL_UINT8(0,   b);
}

static void test_rgb565_pure_green(void) {
    // G=63 (6b) → 0x07E0
    uint8_t r, g, b;
    preprocess_px_rgb565_to_rgb888(0x07E0, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(0,   r);
    TEST_ASSERT_EQUAL_UINT8(255, g);
    TEST_ASSERT_EQUAL_UINT8(0,   b);
}

static void test_rgb565_pure_blue(void) {
    // B=31 (5b) → 0x001F
    uint8_t r, g, b;
    preprocess_px_rgb565_to_rgb888(0x001F, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(0,   r);
    TEST_ASSERT_EQUAL_UINT8(0,   g);
    TEST_ASSERT_EQUAL_UINT8(255, b);
}

static void test_gray_black(void) {
    TEST_ASSERT_EQUAL_UINT8(0, preprocess_rgb_to_gray(0, 0, 0));
}

static void test_gray_white(void) {
    TEST_ASSERT_EQUAL_UINT8(255, preprocess_rgb_to_gray(255, 255, 255));
}

static void test_gray_pure_red(void) {
    // (255*299 + 500) / 1000 = 76.7 → 76
    TEST_ASSERT_EQUAL_UINT8(76, preprocess_rgb_to_gray(255, 0, 0));
}

static void test_gray_pure_green(void) {
    // (255*587 + 500) / 1000 = 150.2 → 150
    TEST_ASSERT_EQUAL_UINT8(150, preprocess_rgb_to_gray(0, 255, 0));
}

static void test_gray_pure_blue(void) {
    // (255*114 + 500) / 1000 = 29.6 → 29
    TEST_ASSERT_EQUAL_UINT8(29, preprocess_rgb_to_gray(0, 0, 255));
}

static void test_normalize_exact_mean(void) {
    // val == mean → (0)*128/std = 0
    TEST_ASSERT_EQUAL_INT8(0, preprocess_normalize_u8(128, 128, 64));
}

static void test_normalize_positive(void) {
    // (192-128)*128/64 = 128 → saturado a 127
    TEST_ASSERT_EQUAL_INT8(127, preprocess_normalize_u8(192, 128, 64));
}

static void test_normalize_negative(void) {
    // (64-128)*128/64 = -128
    TEST_ASSERT_EQUAL_INT8(-128, preprocess_normalize_u8(64, 128, 64));
}

static void test_normalize_saturate_high(void) {
    // (255-0)*128/64 = 510 → 127
    TEST_ASSERT_EQUAL_INT8(127, preprocess_normalize_u8(255, 0, 64));
}

static void test_normalize_saturate_low(void) {
    // (0-255)*128/64 = -510 → -128
    TEST_ASSERT_EQUAL_INT8(-128, preprocess_normalize_u8(0, 255, 64));
}

static void test_sample_nearest_exact(void) {
    /* Imagen 2x2: [R, G; B, W]. RGB565: R=0xF800, G=0x07E0, B=0x001F, W=0xFFFF. */
    uint16_t img[4] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    uint8_t r, g, b;

    preprocess_sample_nearest(img, 2, 2, 0.0f, 0.0f, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(255, r); TEST_ASSERT_EQUAL_UINT8(0, g); TEST_ASSERT_EQUAL_UINT8(0, b);

    preprocess_sample_nearest(img, 2, 2, 1.0f, 0.0f, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(0, r); TEST_ASSERT_EQUAL_UINT8(255, g); TEST_ASSERT_EQUAL_UINT8(0, b);

    preprocess_sample_nearest(img, 2, 2, 0.0f, 1.0f, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(0, r); TEST_ASSERT_EQUAL_UINT8(0, g); TEST_ASSERT_EQUAL_UINT8(255, b);

    preprocess_sample_nearest(img, 2, 2, 1.0f, 1.0f, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(255, r); TEST_ASSERT_EQUAL_UINT8(255, g); TEST_ASSERT_EQUAL_UINT8(255, b);
}

static void test_sample_nearest_clamp(void) {
    uint16_t img[4] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    uint8_t r, g, b;
    /* Coord fuera del rango: clampea. */
    preprocess_sample_nearest(img, 2, 2, -5.0f, -5.0f, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(255, r); TEST_ASSERT_EQUAL_UINT8(0, g); TEST_ASSERT_EQUAL_UINT8(0, b);

    preprocess_sample_nearest(img, 2, 2, 99.0f, 99.0f, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(255, r); TEST_ASSERT_EQUAL_UINT8(255, g); TEST_ASSERT_EQUAL_UINT8(255, b);
}

static void test_sample_bilinear_exact_corner(void) {
    uint16_t img[4] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    uint8_t r, g, b;
    preprocess_sample_bilinear(img, 2, 2, 0.0f, 0.0f, &r, &g, &b);
    TEST_ASSERT_EQUAL_UINT8(255, r); TEST_ASSERT_EQUAL_UINT8(0, g); TEST_ASSERT_EQUAL_UINT8(0, b);
}

static void test_sample_bilinear_midpoint_horizontal(void) {
    /* Entre R (255,0,0) y G (0,255,0): (127 o 128, 127 o 128, 0). */
    uint16_t img[4] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    uint8_t r, g, b;
    preprocess_sample_bilinear(img, 2, 2, 0.5f, 0.0f, &r, &g, &b);
    TEST_ASSERT_INT_WITHIN(1, 127, r);
    TEST_ASSERT_INT_WITHIN(1, 127, g);
    TEST_ASSERT_EQUAL_UINT8(0, b);
}

static void test_sample_bilinear_center(void) {
    /* Centro entre 4 pixeles: promedio simple. */
    uint16_t img[4] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    uint8_t r, g, b;
    preprocess_sample_bilinear(img, 2, 2, 0.5f, 0.5f, &r, &g, &b);
    /* (255 + 0 + 0 + 255) / 4 = 127.5 para R; similar para G y B. */
    TEST_ASSERT_INT_WITHIN(1, 127, r);
    TEST_ASSERT_INT_WITHIN(1, 127, g);
    TEST_ASSERT_INT_WITHIN(1, 127, b);
}

static void test_pipeline_solid_white_gray_identity(void) {
    /* 96×96 blanco → gray 96×96 → todo (255-128)*128/64 = 254 → sat 127 */
    uint16_t src[96*96];
    int8_t   dst[96*96];
    uint8_t  mean[] = {128};
    uint8_t  std[]  = {64};
    fill_solid_rgb565(src, 96*96, 0xFFFF);

    preprocess_rgb565(src, 96, 96, dst, 96, 96, 1, mean, std, true);

    for (int i = 0; i < 96*96; i++) TEST_ASSERT_EQUAL_INT8(127, dst[i]);
}

static void test_pipeline_solid_black_rgb_downsize(void) {
    /* 96×96 negro → rgb 32×32 → todo (0-128)*128/64 = -256 → sat -128 */
    uint16_t src[96*96];
    int8_t   dst[32*32*3];
    uint8_t  mean[] = {128, 128, 128};
    uint8_t  std[]  = {64,  64,  64};
    fill_solid_rgb565(src, 96*96, 0x0000);

    preprocess_rgb565(src, 96, 96, dst, 32, 32, 3, mean, std, true);

    for (int i = 0; i < 32*32*3; i++) TEST_ASSERT_EQUAL_INT8(-128, dst[i]);
}

static void test_pipeline_nearest_vs_bilinear_solid_equal(void) {
    /* Con imagen uniforme los dos modos deben dar lo mismo. */
    uint16_t src[96*96];
    int8_t   a[32*32], b[32*32];
    uint8_t  mean[] = {0};
    uint8_t  std[]  = {128};
    fill_solid_rgb565(src, 96*96, 0x07E0);  /* verde puro */

    preprocess_rgb565(src, 96, 96, a, 32, 32, 1, mean, std, true);
    preprocess_rgb565(src, 96, 96, b, 32, 32, 1, mean, std, false);

    for (int i = 0; i < 32*32; i++) TEST_ASSERT_EQUAL_INT8(a[i], b[i]);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_rgb565_black);
    RUN_TEST(test_rgb565_white);
    RUN_TEST(test_rgb565_pure_red);
    RUN_TEST(test_rgb565_pure_green);
    RUN_TEST(test_rgb565_pure_blue);
    RUN_TEST(test_gray_black);
    RUN_TEST(test_gray_white);
    RUN_TEST(test_gray_pure_red);
    RUN_TEST(test_gray_pure_green);
    RUN_TEST(test_gray_pure_blue);
    RUN_TEST(test_normalize_exact_mean);
    RUN_TEST(test_normalize_positive);
    RUN_TEST(test_normalize_negative);
    RUN_TEST(test_normalize_saturate_high);
    RUN_TEST(test_normalize_saturate_low);
    RUN_TEST(test_sample_nearest_exact);
    RUN_TEST(test_sample_nearest_clamp);
    RUN_TEST(test_sample_bilinear_exact_corner);
    RUN_TEST(test_sample_bilinear_midpoint_horizontal);
    RUN_TEST(test_sample_bilinear_center);
    RUN_TEST(test_pipeline_solid_white_gray_identity);
    RUN_TEST(test_pipeline_solid_black_rgb_downsize);
    RUN_TEST(test_pipeline_nearest_vs_bilinear_solid_equal);
    return UNITY_END();
}

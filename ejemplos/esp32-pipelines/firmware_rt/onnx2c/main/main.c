/* firmware_rt/onnx2c — self-test del runtime onnx2c (C plano, float32).
 *
 * model.c y golden_data.h se instalan desde
 * tools/rt_export/onnx2c/out/<source>/ con tools/install_model.sh (ambos
 * gitignoreados acá). La entrada es float32 NCHW ya normalizada; la
 * referencia son los logits float de onnxruntime en PC sobre el MISMO
 * model.onnx, comparados con tolerancia ONNX2C_TOL. */
#include <stdio.h>

#include "esp_timer.h"
#include "golden_data.h"
#include "selftest_runner.h"

/* Tolerancia float: onnx2c y onnxruntime hacen las mismas cuentas en float32
 * pero en distinto orden; 0.01 en logits es holgado para ruido de redondeo y
 * estricto contra errores reales. */
#define ONNX2C_TOL 0.01f

/* Definida en model.c (generado por onnx2c). */
void entry(const float tensor_input[1][GOLDEN_CH][GOLDEN_H][GOLDEN_W],
           float tensor_logits[1][GOLDEN_NUM_CLASSES]);

static int onnx2c_infer(const void *input, float *logits_out, int n_classes)
{
    float out[1][GOLDEN_NUM_CLASSES];
    int64_t t0 = esp_timer_get_time();
    entry((const float(*)[GOLDEN_CH][GOLDEN_H][GOLDEN_W])input, out);
    int64_t t1 = esp_timer_get_time();
    for (int k = 0; k < n_classes && k < GOLDEN_NUM_CLASSES; k++)
        logits_out[k] = out[0][k];
    return (int)((t1 - t0) / 1000);
}

void app_main(void)
{
    static rt_golden_t goldens[GOLDEN_COUNT];
    static float ref_logits[GOLDEN_COUNT * GOLDEN_NUM_CLASSES];
    for (int i = 0; i < GOLDEN_COUNT; i++) {
        goldens[i].name = GOLDEN_CLASSES[GOLDEN_IMAGES[i].expected];
        goldens[i].data = (const uint8_t *)GOLDEN_IMAGES[i].data;
        for (int k = 0; k < GOLDEN_NUM_CLASSES; k++)
            ref_logits[i * GOLDEN_NUM_CLASSES + k] =
                GOLDEN_IMAGES[i].ref_logits[k];
    }

    printf("onnx2c runtime: input %dx%dx%d float NCHW, %d clases\n",
           GOLDEN_CH, GOLDEN_H, GOLDEN_W, GOLDEN_NUM_CLASSES);

    selftest_uart_loop(goldens, GOLDEN_COUNT, onnx2c_infer, ref_logits,
                       GOLDEN_NUM_CLASSES, GOLDEN_CLASSES, ONNX2C_TOL);
}

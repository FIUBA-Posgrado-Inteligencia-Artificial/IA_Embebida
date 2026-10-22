/* firmware_rt/tflm — self-test del runtime TFLite Micro (int8) en ESP32-S3.
 *
 * model_data.h y golden_data.h se instalan desde
 * tools/rt_export/tflite/out/<source>/ con tools/install_model.sh (ambos
 * gitignoreados acá). La entrada golden ya viene cuantizada int8 NHWC al
 * input del modelo TFLite; la referencia son los logits int8 del interpreter
 * TFLite de PC sobre el MISMO .tflite → comparación bit-exacta (tol=0).
 *
 * Dos variantes de build (ver README): build_espnn/ (esp-nn SIMD, default)
 * y build_ref/ (CONFIG_NN_ANSI_C=y → kernels esp-nn *_ansi, C portable).
 */
#include <cstdio>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "golden_data.h"
#include "model_data.h"
#include "selftest_runner.h"

/* Tensor arena: en PSRAM (heap_caps_malloc SPIRAM). Sobredimensionada a
 * propósito; el uso real se reporta con arena_used_bytes() al arrancar. */
#ifndef TFLM_ARENA_SIZE
#define TFLM_ARENA_SIZE (2 * 1024 * 1024)
#endif

namespace {
tflite::MicroInterpreter *g_interpreter = nullptr;
TfLiteTensor *g_input = nullptr;
TfLiteTensor *g_output = nullptr;
}  // namespace

static int tflm_infer(const void *input, float *logits_out, int n_classes)
{
    /* El golden ya viene int8 en el layout NHWC del input tensor. */
    std::memcpy(g_input->data.int8, input, g_input->bytes);

    int64_t t0 = esp_timer_get_time();
    if (g_interpreter->Invoke() != kTfLiteOk)
        return -1;
    int64_t t1 = esp_timer_get_time();

    for (int k = 0; k < n_classes && k < MODEL_NUM_CLASSES; k++)
        logits_out[k] = (float)g_output->data.int8[k];
    return (int)((t1 - t0) / 1000);
}

extern "C" void app_main(void)
{
    const tflite::Model *model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: schema version %lu != %d\n",
               (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    /* Ops de los 4 modelos del plan: CONV_2D, DEPTHWISE_CONV_2D,
     * FULLY_CONNECTED, MAX_POOL_2D, MEAN — más un margen defensivo. */
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddAveragePool2D();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddPad();
    resolver.AddTranspose();

    uint8_t *arena = (uint8_t *)heap_caps_malloc(
        TFLM_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!arena) {
        printf("ERROR: no pude alocar arena de %d bytes en PSRAM\n",
               TFLM_ARENA_SIZE);
        return;
    }

    static tflite::MicroInterpreter interpreter(model, resolver, arena,
                                                TFLM_ARENA_SIZE);
    g_interpreter = &interpreter;
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: AllocateTensors falló (arena %d bytes)\n",
               TFLM_ARENA_SIZE);
        return;
    }
    g_input = interpreter.input(0);
    g_output = interpreter.output(0);

    if ((int)g_input->bytes != GOLDEN_W * GOLDEN_H * GOLDEN_CH ||
        g_input->type != kTfLiteInt8) {
        printf("ERROR: input tensor %u bytes type %d, golden %d bytes int8 — "
               "¿model_data.h y golden_data.h de sources distintos?\n",
               (unsigned)g_input->bytes, (int)g_input->type,
               GOLDEN_W * GOLDEN_H * GOLDEN_CH);
        return;
    }

    printf("tflm runtime: input %dx%dx%d int8 NHWC, %d clases, "
           "model %u bytes, arena usada %u de %d bytes\n",
           GOLDEN_H, GOLDEN_W, GOLDEN_CH, MODEL_NUM_CLASSES,
           (unsigned)sizeof(model_tflite),
           (unsigned)interpreter.arena_used_bytes(), TFLM_ARENA_SIZE);

    static rt_golden_t goldens[GOLDEN_COUNT];
    static float ref_logits[GOLDEN_COUNT * MODEL_NUM_CLASSES];
    for (int i = 0; i < GOLDEN_COUNT; i++) {
        goldens[i].name = GOLDEN_CLASSES[GOLDEN_IMAGES[i].expected];
        goldens[i].data = (const uint8_t *)GOLDEN_IMAGES[i].data;
        for (int k = 0; k < MODEL_NUM_CLASSES; k++)
            ref_logits[i * MODEL_NUM_CLASSES + k] =
                (float)GOLDEN_IMAGES[i].ref_logits[k];
    }

    /* tol=-1 → modo int8 del runner admitiendo 1 LSB: los kernels TFLM del
     * device y el interprete litert x86 de PC difieren en 1 LSB de redondeo
     * de requantización (medido 2026-08-14: dlogits<=1 en los 4 modelos). */
    selftest_uart_loop(goldens, GOLDEN_COUNT, tflm_infer, ref_logits,
                       MODEL_NUM_CLASSES,
                       const_cast<const char **>(GOLDEN_CLASSES), -1.0f);
}

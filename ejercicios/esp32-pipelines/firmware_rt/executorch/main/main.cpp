/* firmware_rt/executorch — self-test del runtime ExecuTorch portable (float)
 * en ESP32-S3. Carga el .pte embebido (model_pte.h, generado por
 * tools/rt_export/executorch/export_executorch.py), corre inferencias sobre
 * las golden images (golden_data.h) y compara contra los logits de referencia
 * del MISMO runtime en PC, con tolerancia float (tol=0.01).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include "selftest_runner.h"

#include "model_pte.h"    /* model_pte[], MODEL_* (generado, gitignoreado) */
#include "golden_data.h"  /* GOLDEN_* (generado, gitignoreado) */

using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;

#define METHOD_POOL_SIZE (2 * 1024 * 1024)
#define MAX_PLANNED_BUFFERS 16
#define SELFTEST_TOL 0.01f

static uint8_t *alloc_pool(size_t size)
{
    uint8_t *p = (uint8_t *)heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
    if (!p)
        p = (uint8_t *)heap_caps_malloc(size, MALLOC_CAP_8BIT);
    return p;
}

static Program *g_program;
static Method *g_method;

static bool runtime_setup(void)
{
    executorch::runtime::runtime_init();

    static BufferDataLoader loader(model_pte, model_pte_len);
    static Result<Program> program_res = Program::load(&loader);
    if (!program_res.ok()) {
        printf("ERROR: Program::load: 0x%x\n", (int)program_res.error());
        return false;
    }
    g_program = &program_res.get();

    Result<MethodMeta> meta = g_program->method_meta("forward");
    if (!meta.ok()) {
        printf("ERROR: method_meta: 0x%x\n", (int)meta.error());
        return false;
    }

    /* method allocator pool (PSRAM si hay) */
    uint8_t *method_pool = alloc_pool(METHOD_POOL_SIZE);
    if (!method_pool) {
        printf("ERROR: sin memoria para method pool (%d)\n", METHOD_POOL_SIZE);
        return false;
    }
    static MemoryAllocator method_allocator(METHOD_POOL_SIZE, method_pool);
    method_allocator.reset();

    /* planned buffers del .pte */
    static Span<uint8_t> planned_spans[MAX_PLANNED_BUFFERS];
    size_t n_planned = meta->num_memory_planned_buffers();
    if (n_planned > MAX_PLANNED_BUFFERS) {
        printf("ERROR: %d planned buffers > max %d\n", (int)n_planned,
               MAX_PLANNED_BUFFERS);
        return false;
    }
    for (size_t i = 0; i < n_planned; i++) {
        size_t sz = (size_t)meta->memory_planned_buffer_size(i).get();
        uint8_t *buf = alloc_pool(sz);
        if (!buf) {
            printf("ERROR: sin memoria para planned buffer %d (%d bytes)\n",
                   (int)i, (int)sz);
            return false;
        }
        printf("planned buffer %d: %d bytes\n", (int)i, (int)sz);
        planned_spans[i] = {buf, sz};
    }
    static HierarchicalAllocator planned_memory(
        {planned_spans, n_planned});
    static MemoryManager memory_manager(&method_allocator, &planned_memory);

    static Result<Method> method_res =
        g_program->load_method("forward", &memory_manager);
    if (!method_res.ok()) {
        printf("ERROR: load_method: 0x%x\n", (int)method_res.error());
        return false;
    }
    g_method = &method_res.get();
    printf("modelo cargado: pte=%d bytes, input %dx%dx%d, %d clases\n",
           (int)model_pte_len, MODEL_INPUT_CHANNELS, MODEL_INPUT_H,
           MODEL_INPUT_W, MODEL_NUM_CLASSES);
    return true;
}

/* rt_infer_fn: input = float NCHW normalizado (golden_data.h) */
static int et_infer(const void *input, float *logits_out, int n_classes)
{
    static TensorImpl::SizesType sizes[4] = {
        1, MODEL_INPUT_CHANNELS, MODEL_INPUT_H, MODEL_INPUT_W};
    static TensorImpl::DimOrderType dim_order[4] = {0, 1, 2, 3};
    static TensorImpl::StridesType strides[4] = {
        MODEL_INPUT_CHANNELS * MODEL_INPUT_H * MODEL_INPUT_W,
        MODEL_INPUT_H * MODEL_INPUT_W, MODEL_INPUT_W, 1};
    TensorImpl impl(ScalarType::Float, 4, sizes,
                    const_cast<void *>(input), dim_order, strides);
    Tensor t(&impl);

    Error err = g_method->set_input(t, 0);
    if (err != Error::Ok) return -1;

    int64_t t0 = esp_timer_get_time();
    err = g_method->execute();
    int64_t t1 = esp_timer_get_time();
    if (err != Error::Ok) return -2;

    const auto &out = g_method->get_output(0);
    if (!out.isTensor()) return -3;
    const Tensor &ot = out.toTensor();
    const float *data = ot.const_data_ptr<float>();
    int n = (int)ot.numel();
    if (n > n_classes) n = n_classes;
    for (int k = 0; k < n; k++) logits_out[k] = data[k];
    return (int)((t1 - t0) / 1000);
}

extern "C" void app_main(void)
{
    printf("\n=== firmware_rt/executorch (portable, float) ===\n");

    if (!runtime_setup()) {
        printf("FATAL: setup del runtime falló\n");
        vTaskDelay(portMAX_DELAY);
    }

    /* Aplanar golden data al formato del selftest_runner */
    static rt_golden_t imgs[GOLDEN_COUNT];
    static float ref_logits[GOLDEN_COUNT * MODEL_NUM_CLASSES];
    static const char *class_names[MODEL_NUM_CLASSES];
    for (int i = 0; i < GOLDEN_COUNT; i++) {
        imgs[i].name = GOLDEN_CLASSES[GOLDEN_IMAGES[i].expected];
        imgs[i].data = (const uint8_t *)GOLDEN_IMAGES[i].data;
        memcpy(&ref_logits[i * MODEL_NUM_CLASSES], GOLDEN_IMAGES[i].ref_logits,
               MODEL_NUM_CLASSES * sizeof(float));
    }
    for (int k = 0; k < MODEL_NUM_CLASSES; k++)
        class_names[k] = MODEL_CLASSES[k];

    selftest_uart_loop(imgs, GOLDEN_COUNT, et_infer, ref_logits,
                       MODEL_NUM_CLASSES, class_names, SELFTEST_TOL);
}

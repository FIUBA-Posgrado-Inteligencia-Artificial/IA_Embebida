#include "inference.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <cmath>
#include <cstring>

/* ESP-DL v3 public API header (replaces dl_model.hpp from v2) */
#include "dl_model_base.hpp"

static const char *TAG = "inference";
static bool       g_dummy_mode = false;
static dl::Model *g_model      = nullptr;

extern "C" esp_err_t inference_init(void)
{
    if (model_data_len == 0) {
        ESP_LOGW(TAG, "empty model, dummy mode");
        g_dummy_mode = true;
        return ESP_OK;
    }
    /*
     * ESP-DL v3 API adaptation (v2 used dl::Model(const void*, size_t)).
     * v3 constructor takes (const char* rodata_addr_or_label, location).
     * Casting the uint8_t rodata array to const char* is the documented
     * pattern for MODEL_LOCATION_IN_FLASH_RODATA with a C array.
     * Exception handling (-fexceptions) is off in IDF by default, so we
     * allocate without try/catch and guard via a nullptr check instead.
     */
    g_model = new (std::nothrow) dl::Model(
        reinterpret_cast<const char *>(model_data),
        fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    if (g_model == nullptr) {
        ESP_LOGE(TAG, "model alloc failed, dummy mode");
        g_dummy_mode = true;
        return ESP_OK;
    }
    ESP_LOGI(TAG, "ESP-DL model loaded, %u bytes", (unsigned)model_data_len);
    return ESP_OK;
}

extern "C" bool inference_is_dummy_mode(void) { return g_dummy_mode; }

static void softmax_and_entropy(const float *logits, int n,
                                float *probs, float *entropy_norm)
{
    float zmax = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > zmax) zmax = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - zmax);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;

    float H = 0.0f;
    for (int i = 0; i < n; i++) {
        float p = probs[i] < 1e-12f ? 1e-12f : probs[i];
        H -= p * logf(p);
    }
    *entropy_norm = H / logf((float)n);
}

extern "C" esp_err_t inference_run(const int8_t *input, inference_result_t *r)
{
    int64_t t0 = esp_timer_get_time();
    float logits[MODEL_NUM_CLASSES];

    if (g_dummy_mode) {
        /* Uniform logits → softmax yields 1/N each → H_normalized = 1.0 */
        for (int i = 0; i < MODEL_NUM_CLASSES; i++) logits[i] = 0.0f;
    } else if (g_model != nullptr) {
        /*
         * ESP-DL v3 API adaptation:
         *   - get_inputs() / get_outputs() return std::map<std::string, TensorBase*>&
         *   - TensorBase::get_element_ptr() returns void*
         *   - TensorBase::exponent is ExponentInfo with operator int() for compat
         * TODO-BINDING: if the model output dtype is not int8 (e.g. float),
         * adjust the dequant logic below accordingly.
         */
        dl::Model *m = g_model; /* local non-null pointer satisfies -Wnonnull */
        auto &inputs = m->get_inputs();
        dl::TensorBase *in_tensor = inputs.begin()->second;
        std::memcpy(in_tensor->get_element_ptr(), input,
                    MODEL_INPUT_W * MODEL_INPUT_H * MODEL_INPUT_CHANNELS);

        m->run();

        auto &outputs = m->get_outputs();
        dl::TensorBase *out = outputs.begin()->second;
        int8_t *out_data = static_cast<int8_t *>(out->get_element_ptr());
        int exp_val = static_cast<int>(out->exponent); /* ExponentInfo → int */
        float scale = exp_val < 0
                    ? 1.0f / (float)(1 << -exp_val)
                    : (float)(1 << exp_val);
        for (int i = 0; i < MODEL_NUM_CLASSES; i++) logits[i] = out_data[i] * scale;
    } else {
        /* Should not happen: g_dummy_mode guards this, but belt-and-suspenders */
        for (int i = 0; i < MODEL_NUM_CLASSES; i++) logits[i] = 0.0f;
    }

    softmax_and_entropy(logits, MODEL_NUM_CLASSES, r->probs, &r->entropy_normalized);

    r->top1_idx        = 0;
    r->top1_confidence = r->probs[0];
    for (int i = 1; i < MODEL_NUM_CLASSES; i++) {
        if (r->probs[i] > r->top1_confidence) {
            r->top1_confidence = r->probs[i];
            r->top1_idx        = i;
        }
    }
    r->inference_us = esp_timer_get_time() - t0;
    return ESP_OK;
}

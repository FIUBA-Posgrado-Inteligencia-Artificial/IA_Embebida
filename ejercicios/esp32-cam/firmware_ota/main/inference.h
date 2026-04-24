#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
#include "model_data.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float    probs[MODEL_NUM_CLASSES];
    int      top1_idx;
    float    top1_confidence;
    float    entropy_normalized;
    int64_t  inference_us;
} inference_result_t;

esp_err_t inference_init(void);
esp_err_t inference_run(const int8_t *input, inference_result_t *result);
bool      inference_is_dummy_mode(void);

#ifdef __cplusplus
}
#endif

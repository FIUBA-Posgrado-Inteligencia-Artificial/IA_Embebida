#include "result_format.h"
#include "model_data.h"
#include "sdkconfig.h"
#include "cJSON.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool is_low_conf(const inference_result_t *r) {
    int pct = (int)(r->entropy_normalized * 100.0f + 0.5f);
    return pct >= CONFIG_ESP32CAM_ENTROPY_WARN_THRESHOLD;
}

void result_format_text(const inference_result_t *r, char *buf, size_t n)
{
    const char *prefix = is_low_conf(r) ? "[LOW_CONFIDENCE]" : "";
    const char *top1 = (r->top1_idx >= 0 && r->top1_idx < MODEL_NUM_CLASSES)
                     ? MODEL_CLASSES[r->top1_idx] : "?";
    snprintf(buf, n, "%s[top1=%s conf=%.3f H=%.3f t=%lldms]",
             prefix, top1, r->top1_confidence, r->entropy_normalized,
             (long long)(r->inference_us / 1000));
}

void result_format_json(const inference_result_t *r, char *buf, size_t n)
{
    cJSON *root = cJSON_CreateObject();
    cJSON *classes = cJSON_CreateObject();
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        cJSON_AddNumberToObject(classes, MODEL_CLASSES[i], r->probs[i]);
    }
    cJSON_AddItemToObject(root, "classes", classes);
    const char *top1_name = (r->top1_idx >= 0 && r->top1_idx < MODEL_NUM_CLASSES)
                          ? MODEL_CLASSES[r->top1_idx] : "?";
    cJSON_AddStringToObject(root, "top1", top1_name);
    cJSON_AddNumberToObject(root, "confidence", r->top1_confidence);
    cJSON_AddNumberToObject(root, "entropy_normalized", r->entropy_normalized);
    cJSON_AddNumberToObject(root, "inference_ms", (double)(r->inference_us / 1000));
    cJSON_AddBoolToObject(root, "low_confidence", is_low_conf(r));

    char *printed = cJSON_PrintUnformatted(root);
    if (printed) {
        strncpy(buf, printed, n - 1);
        buf[n - 1] = '\0';
        cJSON_free(printed);
    } else if (n > 0) {
        buf[0] = '\0';
    }
    cJSON_Delete(root);
}

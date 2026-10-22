#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "sdkconfig.h"

#include "board.h"
#include "camera.h"
#include "preprocess.h"
#include "inference.h"
#include "result_format.h"
#include "serial_trigger.h"
#include "wifi_manager.h"
#include "esp32cam_mqtt.h"
#include "http_stream_server.h"
#include "trigger.h"
#include "model_data.h"
#include "capture_state.h"
#include "esp_timer.h"
#include "img_converters.h"

#if !BOARD_HAS_CAMERA
#include "static_frame.h"
#endif

/* Golden images para self-test (tecla 't'). El header lo genera
 * gen_test_images.py y está gitignoreado: si no existe, el self-test
 * queda deshabilitado y el firmware compila igual. */
#if FEATURE_HW_VALIDATION
# if defined(__has_include)
#  if __has_include("test_images.h")
#   include "test_images.h"
#   define HAVE_TEST_IMAGES 1
#  endif
# endif
#endif

static const char *TAG = "app_main";

static int8_t g_input_buf[MODEL_INPUT_W * MODEL_INPUT_H * MODEL_INPUT_CHANNELS];

/* Self-test: inferencia sobre las golden images embebidas (sin cámara ni
 * preprocess). Si el device coincide con lo medido en PC, el modelo y la
 * cuantización están sanos en hardware. */
static void run_self_test(void)
{
#if !FEATURE_HW_VALIDATION
    printf("self-test deshabilitado por config (FEATURE_HW_VALIDATION=0)\n");
#elif defined(HAVE_TEST_IMAGES)
    _Static_assert(TEST_IMAGES_W == MODEL_INPUT_W &&
                   TEST_IMAGES_H == MODEL_INPUT_H &&
                   TEST_IMAGES_CH == MODEL_INPUT_CHANNELS,
                   "test_images.h no corresponde al modelo instalado — "
                   "regenerar con gen_test_images.py y rebuildear");
    int pass = 0;
    printf("\n=== SELF-TEST: %d golden images (STL-10 test set) ===\n",
           TEST_IMAGES_COUNT);
    printf("  PASS = coincide con el modelo CUANTIZADO simulado en PC "
           "(esperable %d/%d exacto)\n", TEST_IMAGES_COUNT, TEST_IMAGES_COUNT);
    for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
        inference_result_t r;
        int64_t t0 = esp_timer_get_time();
        if (inference_run(TEST_IMAGES[i].data, &r) != ESP_OK) {
            printf("  %-9s -> ERROR de inferencia\n",
                   MODEL_CLASSES[TEST_IMAGES[i].expected]);
            continue;
        }
        uint32_t ms = (uint32_t)((esp_timer_get_time() - t0) / 1000);
        /* PASS estricto: contra la predicción del modelo cuantizado en PC.
         * La clase verdadera y la conf float se imprimen como contexto. */
        bool ok = (r.top1_idx == TEST_IMAGES[i].pc_q_top1);
        if (ok) pass++;

        /* Paridad a nivel logits int8: cuántos LSB difiere el device del sim */
        int max_lsb = 0;
        for (int k = 0; k < MODEL_NUM_CLASSES; k++) {
            int d = (int)r.raw_logits[k] - (int)TEST_IMAGES[i].pc_q_logits[k];
            if (d < 0) d = -d;
            if (d > max_lsb) max_lsb = d;
        }

        printf("  %-9s -> %-9s %5.1f%%  (pc_q: %-9s %5.1f%% | float %5.1f%%)  "
               "dlogits<=%d  %ums  %s\n",
               MODEL_CLASSES[TEST_IMAGES[i].expected],
               MODEL_CLASSES[r.top1_idx],
               r.top1_confidence * 100.0f,
               MODEL_CLASSES[TEST_IMAGES[i].pc_q_top1],
               TEST_IMAGES[i].pc_q_conf * 100.0f,
               TEST_IMAGES[i].pc_conf * 100.0f,
               max_lsb,
               (unsigned)ms,
               ok ? "PASS" : "FAIL");
        if (max_lsb > 0) {
            printf("      logits dev:");
            for (int k = 0; k < MODEL_NUM_CLASSES; k++)
                printf(" %4d", (int)r.raw_logits[k]);
            printf("\n      logits pc :");
            for (int k = 0; k < MODEL_NUM_CLASSES; k++)
                printf(" %4d", (int)TEST_IMAGES[i].pc_q_logits[k]);
            printf("\n");
        }
    }
    printf("=== SELF-TEST: %d/%d PASS (vs PC cuantizado) ===\n"
           "    dlogits = max |device - sim| en LSB int8: 0 = bit-exacto\n\n",
           pass, TEST_IMAGES_COUNT);
#else
    printf("self-test no disponible: falta test_images.h — generarlo con\n"
           "  uv run python gen_test_images.py --source <checkpoint>\n"
           "y rebuildear.\n");
#endif
}

/* Reporte legible por serie: encabezado, tiempos y barras por clase. */
static void print_result_pretty(uint32_t seq, trigger_source_t src,
                                const inference_result_t *r,
                                uint32_t cap_ms, uint32_t pre_ms,
                                uint32_t inf_ms, float fps)
{
    static const char BAR[] = "########################";  /* 24 chars */
    const char *src_s = src == TRIGGER_SERIAL ? "serial"
                      : src == TRIGGER_HTTP   ? "http" : "mqtt";

    /* top-3 por selección directa (N=10, no vale la pena ordenar todo) */
    int top[3] = { -1, -1, -1 };
    for (int i = 0; i < MODEL_NUM_CLASSES; i++)
        for (int k = 0; k < 3; k++)
            if (top[k] < 0 || r->probs[i] > r->probs[top[k]]) {
                for (int j = 2; j > k; j--) top[j] = top[j - 1];
                top[k] = i;
                break;
            }

    printf("\n=== captura #%u (%s) ===\n", (unsigned)seq, src_s);
    printf("  top3: ");
    for (int k = 0; k < 3; k++)
        printf("%s%s %.1f%%", k ? "  |  " : "",
               MODEL_CLASSES[top[k]], r->probs[top[k]] * 100.0f);
    printf("   entropia: %.2f%s\n", r->entropy_normalized,
           r->entropy_normalized >= 0.7f ? "  [BAJA CONFIANZA]" : "");
    printf("  tiempos: captura %ums | preproceso %ums | inferencia %ums | total %ums",
           (unsigned)cap_ms, (unsigned)pre_ms, (unsigned)inf_ms,
           (unsigned)(cap_ms + pre_ms + inf_ms));
    if (fps > 0) printf(" | %.1f fps", fps);
    printf("\n");
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        int n = (int)(r->probs[i] * (float)(sizeof BAR - 1) + 0.5f);
        printf("  %c %-9s %5.1f%% %.*s\n",
               i == r->top1_idx ? '>' : ' ',
               MODEL_CLASSES[i], r->probs[i] * 100.0f, n, BAR);
    }
}

/* Devuelve true si hubo resultado. fps<=0 → modo single (no se reporta). */
static bool run_pipeline(trigger_source_t src, float fps)
{
    /* Progreso por etapas solo en modo single: en continuo (~fps alto) el
     * bloque de resultado ya es suficiente y esto seria puro ruido. */
    const bool verbose = (fps <= 0.0f);
    int64_t t0 = esp_timer_get_time();
    if (verbose) { printf("[%s] captura...", src == TRIGGER_SERIAL ? "c" :
                          src == TRIGGER_HTTP ? "http" : "mqtt"); fflush(stdout); }
    const uint16_t *buf; int actual_w, actual_h;
#if BOARD_HAS_CAMERA
    camera_fb_t *fb = NULL;
    if (camera_capture(&fb) != ESP_OK || fb == NULL) {
        if (verbose) printf(" ERROR\n");
        capture_state_publish_error("camera_capture failed");
        return false;
    }
    /* fb->height is the configured dimension; the OV2640 DVP timing can deliver
     * fewer lines (always an exact half-buffer period short). Derive the actual
     * height from fb->len so the resize is correct regardless. */
    actual_w = (int)fb->width;
    actual_h = (fb->format == PIXFORMAT_RGB565 && actual_w > 0)
               ? (int)fb->len / (actual_w * 2) : (int)fb->height;
    buf = (const uint16_t *)fb->buf;

    /* JPEG para /last.jpg: encodear el RGB565 recién capturado. Se publica
     * junto con el resultado (atómico) recién después de la inferencia. */
    uint8_t *jpg = NULL; size_t jpg_len = 0;
    if (!fmt2jpg(fb->buf, fb->len, actual_w, actual_h, PIXFORMAT_RGB565,
                 80, &jpg, &jpg_len)) {
        jpg = NULL; jpg_len = 0;
    }
#else
    buf = STATIC_FRAME; actual_w = STATIC_FRAME_W; actual_h = STATIC_FRAME_H;

    /* No hay captura real: encodeamos el frame fijo a JPEG cada vez para que
     * /last.jpg tenga contenido (no hay t0 de captura que descontar). */
    uint8_t *jpg = NULL; size_t jpg_len = 0;
    if (!fmt2jpg((uint8_t *)buf, (size_t)actual_w * actual_h * 2, actual_w,
                 actual_h, PIXFORMAT_RGB565, 80, &jpg, &jpg_len)) {
        jpg = NULL; jpg_len = 0;
    }
    t0 = esp_timer_get_time();   /* no hay captura real: cap_ms reporta 0 */
#endif
    int64_t t1 = esp_timer_get_time();   /* fin captura+jpeg */
    if (verbose) { printf(" %u ms\n    preproceso...", (unsigned)((t1 - t0) / 1000)); fflush(stdout); }

    uint8_t mean[3] = { MODEL_INPUT_MEAN_0,
#if MODEL_INPUT_CHANNELS == 3
                        MODEL_INPUT_MEAN_1, MODEL_INPUT_MEAN_2
#else
                        0, 0
#endif
                      };
    uint8_t std[3]  = { MODEL_INPUT_STD_0,
#if MODEL_INPUT_CHANNELS == 3
                        MODEL_INPUT_STD_1, MODEL_INPUT_STD_2
#else
                        1, 1
#endif
                      };
    preprocess_rgb565(buf, actual_w, actual_h, g_input_buf,
                      MODEL_INPUT_W, MODEL_INPUT_H, MODEL_INPUT_CHANNELS,
                      mean, std,
                      RESIZE_BILINEAR ? true : false,
                      1 << -(MODEL_INPUT_EXPONENT));
#if BOARD_HAS_CAMERA
    camera_return(fb);
#endif
    int64_t t2 = esp_timer_get_time();   /* fin preprocess */
    if (verbose) { printf(" %u ms\n    inferencia...", (unsigned)((t2 - t1) / 1000)); fflush(stdout); }

    inference_result_t result;
    inference_run(g_input_buf, &result);
    int64_t t3 = esp_timer_get_time();   /* fin inferencia */
    if (verbose) printf(" %u ms\n", (unsigned)((t3 - t2) / 1000));

    uint32_t cap_ms = (uint32_t)((t1 - t0) / 1000);
    uint32_t pre_ms = (uint32_t)((t2 - t1) / 1000);
    uint32_t inf_ms = (uint32_t)((t3 - t2) / 1000);
    uint32_t seq = capture_state_publish_result(&result, jpg, jpg_len,
                                                cap_ms, pre_ms, inf_ms,
                                                fps > 0 ? fps : 0.0f);
    free(jpg);

    print_result_pretty(seq, src, &result, cap_ms, pre_ms, inf_ms, fps);

#ifdef CONFIG_ESP32CAM_MQTT_ENABLED
    char json[768];
    result_format_json(&result, json, sizeof json);
    esp32cam_mqtt_publish_result(json);
#endif
    return true;
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

#if BOARD_HAS_CAMERA
    if (camera_init() != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed, rebooting in 3s");
        vTaskDelay(pdMS_TO_TICKS(3000));
        esp_restart();
    }
#else
    ESP_LOGW(TAG, "cámara deshabilitada (BOARD_HAS_CAMERA=0) — "
                  "inferencia sobre imagen fija embebida (%s)",
             STATIC_FRAME_CLASS);
#endif
    inference_init();
    if (capture_state_init() != ESP_OK) {
        ESP_LOGE(TAG, "capture state init failed, rebooting in 3s");
        vTaskDelay(pdMS_TO_TICKS(3000));
        esp_restart();
    }
    if (inference_is_dummy_mode()) {
        ESP_LOGW(TAG, "running in dummy inference mode");
    }

    QueueHandle_t trigger_queue = xQueueCreate(4, sizeof(trigger_source_t));
    serial_trigger_start(trigger_queue);

#if FEATURE_WIFI
    /* Cold boot note: PHY init reliably crashes on the first boot after a
     * full power-cycle (3.3V rail transient during RF activation). The
     * firmware auto-recovers on the next boot (PSRAM caps stay charged,
     * so warm reset succeeds). Net effect: ~2s longer first-boot latency
     * after a power-cycle; zero impact during dev iterations. We don't
     * try to prevent it in software — we just let it happen. */
    wifi_manager_start();
#endif
#ifdef CONFIG_ESP32CAM_MQTT_ENABLED
    esp32cam_mqtt_start(trigger_queue);
#endif
#if FEATURE_WEBSERVER
    http_stream_server_start(trigger_queue);
#endif

    ESP_LOGI(TAG, "ready — any byte = capture, 'm' = toggle mode, 't' = self-test / MQTT: publish to %s",
#ifdef CONFIG_ESP32CAM_MQTT_ENABLED
             CONFIG_ESP32CAM_MQTT_TOPIC_CAPTURE
#else
             "(disabled)"
#endif
    );
    serial_trigger_print_help();

    trigger_source_t src;
    float fps = 0.0f;
    while (1) {
        if (capture_state_get_mode() == CAPTURE_MODE_CONTINUOUS) {
            int64_t it0 = esp_timer_get_time();
            /* drenar triggers pendientes para que no se acumulen
             * (el self-test sí se atiende, aun en continuo) */
            while (xQueueReceive(trigger_queue, &src, 0) == pdTRUE) {
                if (src == TRIGGER_SELFTEST) run_self_test();
            }
            if (run_pipeline(TRIGGER_SERIAL, fps)) {
                int64_t dt_us = esp_timer_get_time() - it0;
                float inst = dt_us > 0 ? 1e6f / (float)dt_us : 0.0f;
                fps = (fps <= 0.0f) ? inst : (0.7f * fps + 0.3f * inst);
            } else {
                vTaskDelay(pdMS_TO_TICKS(200));
            }
        } else {
            fps = 0.0f;
            /* timeout corto para reaccionar si el modo cambia a continuo */
            if (xQueueReceive(trigger_queue, &src, pdMS_TO_TICKS(250)) == pdTRUE) {
                if (src == TRIGGER_SELFTEST) run_self_test();
                else run_pipeline(src, 0.0f);
            }
        }
    }
}

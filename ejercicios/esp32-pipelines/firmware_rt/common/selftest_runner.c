/* Ver selftest_runner.h. Estilo/formato de salida copiado de
 * firmware/main/app_main.c (run_self_test) y serial_trigger.c. */
#include "selftest_runner.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "driver/uart.h"

#define ST_UART_NUM   UART_NUM_0
#define ST_MAX_CLASSES 32

/* Softmax top1 en float sobre los logits del device (numéricamente estable). */
static float softmax_top1(const float *logits, int n, int *top1_idx)
{
    int best = 0;
    for (int k = 1; k < n; k++)
        if (logits[k] > logits[best]) best = k;
    float denom = 0.0f;
    for (int k = 0; k < n; k++)
        denom += expf(logits[k] - logits[best]);
    *top1_idx = best;
    return 1.0f / denom;
}

static int argmax(const float *v, int n)
{
    int best = 0;
    for (int k = 1; k < n; k++)
        if (v[k] > v[best]) best = k;
    return best;
}

int selftest_run(const rt_golden_t *imgs, int n, rt_infer_fn infer,
                 const float *ref_logits, int n_classes,
                 const char **class_names, float tol)
{
    float logits[ST_MAX_CLASSES];
    if (n_classes > ST_MAX_CLASSES) {
        printf("self-test: n_classes=%d > max %d\n", n_classes, ST_MAX_CLASSES);
        return 0;
    }

    int pass = 0;
    printf("\n=== SELF-TEST: %d golden images ===\n", n);
    for (int i = 0; i < n; i++) {
        const float *ref = &ref_logits[i * n_classes];
        int ms = infer(imgs[i].data, logits, n_classes);
        if (ms < 0) {
            printf("  %-9s -> ERROR de inferencia (%d)\n", imgs[i].name, ms);
            continue;
        }

        int top1;
        float conf = softmax_top1(logits, n_classes, &top1);
        int ref_top1 = argmax(ref, n_classes);

        float maxd = 0.0f;
        for (int k = 0; k < n_classes; k++) {
            float d = fabsf(logits[k] - ref[k]);
            if (d > maxd) maxd = d;
        }

        int dlogits;
        int within;
        if (tol <= 0.0f) {
            /* modo int8: delta entero en LSB. tol=0 exige bit-exacto;
             * tol=-N admite hasta N LSB (p.ej. TFLM device vs interprete
             * x86 de PC difieren en 1 LSB por redondeo de requant). */
            int allowed = (int)lroundf(-tol);
            dlogits = (int)lroundf(maxd);
            within = (dlogits <= allowed);
        } else {
            /* modo float: 0 si dentro de tolerancia, sino escalado a entero */
            within = (maxd < tol);
            dlogits = within ? 0 : (int)ceilf(maxd / tol);
        }

        int ok = (top1 == ref_top1) && within;
        if (ok) pass++;

        /* Formato parseado por script06-bench.py — no cambiar. */
        printf("  %-9s -> %-9s %5.1f%%  dlogits<=%d  %dms  %s\n",
               imgs[i].name, class_names[top1], conf * 100.0f,
               dlogits, ms, ok ? "PASS" : "FAIL");
    }
    printf("self-test: %d/%d PASS\n\n", pass, n);
    return pass;
}

static void print_help(void)
{
    printf("\n--- opciones (terminal) ---\n"
           "  c : inferencia sobre una golden image\n"
           "  t : self-test (todas las golden images)\n"
           "  h : esta ayuda\n"
           "  (el resto de las teclas se ignora)\n\n");
}

void selftest_uart_loop(const rt_golden_t *imgs, int n, rt_infer_fn infer,
                        const float *ref_logits, int n_classes,
                        const char **class_names, float tol)
{
    /* UART0 ya lo inicializó la consola del IDF; sólo instalamos driver RX
     * (mismo patrón que firmware/main/serial_trigger.c). */
    esp_err_t err = uart_driver_install(ST_UART_NUM, 256, 0, 0, NULL, 0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE)
        printf("selftest_uart_loop: uart_driver_install: %d\n", (int)err);

    printf("ready\n");
    print_help();

    while (1) {
        uint8_t byte;
        int got = uart_read_bytes(ST_UART_NUM, &byte, 1, portMAX_DELAY);
        if (got <= 0) continue;
        /* drena buffer pendiente para consolidar en un único comando */
        uart_flush_input(ST_UART_NUM);

        switch (byte) {
        case 'c': case 'C': {
            if (n <= 0) break;
            float logits[ST_MAX_CLASSES];
            if (n_classes > ST_MAX_CLASSES) break;
            int ms = infer(imgs[0].data, logits, n_classes);
            if (ms < 0) {
                printf("inferencia: ERROR (%d)\n", ms);
                break;
            }
            int top1;
            float conf = softmax_top1(logits, n_classes, &top1);
            printf("  %-9s -> %-9s %5.1f%%  %dms\n",
                   imgs[0].name, class_names[top1], conf * 100.0f, ms);
            break;
        }
        case 't': case 'T':
            selftest_run(imgs, n, infer, ref_logits, n_classes,
                         class_names, tol);
            break;
        case 'h': case 'H': case '?':
            print_help();
            break;
        default:
            /* teclas desconocidas se ignoran (h = ayuda) */
            break;
        }
    }
}

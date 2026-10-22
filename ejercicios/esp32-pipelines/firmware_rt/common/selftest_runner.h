/* selftest_runner — self-test compartido entre los firmwares de runtimes
 * alternativos (firmware_rt/{tflm,onnx2c,executorch}).
 *
 * Cada firmware provee sólo su función de inferencia (rt_infer_fn) y los
 * datos golden (input YA preprocesado al formato del runtime + logits de
 * referencia medidos en PC con el MISMO runtime). Este componente imprime
 * las líneas que parsea script06-bench.py:
 *
 *   ^\s*(\w+)\s+->\s+(\w+)\s+([\d.]+)%.*dlogits<=(\d+)\s+(\d+)ms\s+(PASS|FAIL)
 *
 * p.ej.:  "  airplane  -> airplane   93.2%  dlogits<=0  11ms  PASS"
 */
#ifndef SELFTEST_RUNNER_H
#define SELFTEST_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Una golden image: nombre de la clase verdadera + input preprocesado
 * (int8/float/lo que consuma el runtime; el runner no lo interpreta). */
typedef struct {
    const char *name;
    const uint8_t *data;
} rt_golden_t;

/* Corre una inferencia sobre `input` y escribe n_classes logits float en
 * `logits_out` (para runtimes int8, castear el logit int8 a float tal cual).
 * Devuelve el tiempo de inferencia en ms, o <0 en error. */
typedef int (*rt_infer_fn)(const void *input, float *logits_out, int n_classes);

/* Corre el self-test completo e imprime una línea por imagen + resumen
 * "self-test: N/M PASS".
 *
 * ref_logits: n * n_classes floats, logits de referencia de PC (para
 *   runtimes int8 son los logits int8 casteados a float — enteros exactos).
 * tol: selecciona el modo de comparación de logits:
 *   - tol <= 0 (runtimes int8): dlogits = max|device - ref| redondeado a
 *     entero (LSB int8). PASS exige dlogits <= -tol (tol=0: bit-exacto;
 *     tol=-1: admite 1 LSB de diferencia de redondeo, etc.).
 *   - tol > 0 (runtimes float): dlogits = 0 si max|Δlogit| < tol, sino el
 *     Δ escalado a entero, ceil(maxΔ/tol). PASS exige max|Δlogit| < tol.
 * En ambos modos PASS además exige que la predicción (argmax device) coincida
 * con la referencia (argmax ref_logits).
 *
 * Devuelve la cantidad de PASS. */
int selftest_run(const rt_golden_t *imgs, int n, rt_infer_fn infer,
                 const float *ref_logits, int n_classes,
                 const char **class_names, float tol);

/* Loop bloqueante sobre UART0: imprime "ready" + mini menú y atiende
 *   c : una inferencia sobre la primera golden image
 *   t : self-test completo (selftest_run)
 *   h : ayuda
 * El resto de las teclas se ignora. No retorna. */
void selftest_uart_loop(const rt_golden_t *imgs, int n, rt_infer_fn infer,
                        const float *ref_logits, int n_classes,
                        const char **class_names, float tol);

#ifdef __cplusplus
}
#endif

#endif /* SELFTEST_RUNNER_H */

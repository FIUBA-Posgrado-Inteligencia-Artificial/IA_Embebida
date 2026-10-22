#pragma once
#include <stddef.h>
#include "inference.h"

/* Salida humana en una línea, ej:
 *   [top1=bird conf=0.71 H=0.18 t=45ms]
 *   [LOW_CONFIDENCE][top1=bird ...]   si H*100 >= threshold
 */
void result_format_text(const inference_result_t *r, char *buf, size_t n);

/* JSON compacto (cJSON). Mismo schema que el spec Sección 10. */
void result_format_json(const inference_result_t *r, char *buf, size_t n);

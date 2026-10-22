/* firmware_config.h — configuración de features, independiente de la placa.
 * No incluir directo: siempre via board.h. */
#pragma once

#define FEATURE_WIFI 1              /* habilita el stack WiFi (STA) */
#define WIFI_SSID "TeleCentro-fecf" /* SSID de la red a la que conectar */
#define WIFI_PASSWORD "sinlugaradudas" /* password de la red (vacío = abierta) */
#define FEATURE_WEBSERVER 1         /* habilita el servidor HTTP (requiere FEATURE_WIFI) */
#define FEATURE_HW_VALIDATION 1     /* habilita chequeos de hardware al boot */
#define FEATURE_PSRAM -1            /* -1 = auto (usa BOARD_HAS_PSRAM), 0/1 = forzar */
#define HTTP_PORT 80                /* puerto del servidor HTTP */
#define RESIZE_BILINEAR 1           /* 1 = resize bilineal, 0 = nearest */
#define START_IN_CONTINUOUS 0       /* 1 = arrancar en modo captura continua */
#define ENTROPY_WARN_THRESHOLD 70   /* umbral (%) de entropía para warning */

/* Dónde viven los PESOS del modelo (experimento didáctico de jerarquía de
 * memoria; ver README § "Efecto de la memoria"). AUTO = comportamiento
 * normal (param_copy a PSRAM si hay). Las otras opciones copian el blob a la
 * memoria pedida y leen los pesos de ahí (param_copy off). */
#define MODEL_PLACEMENT_AUTO 0
#define MODEL_PLACEMENT_FLASH 1     /* rodata mapeada por caché (la más lenta) */
#define MODEL_PLACEMENT_PSRAM 2     /* copia a PSRAM (requiere PSRAM funcional) */
#define MODEL_PLACEMENT_INTERNAL 3  /* copia a SRAM interna (la más rápida) */
#define MODEL_PLACEMENT MODEL_PLACEMENT_AUTO

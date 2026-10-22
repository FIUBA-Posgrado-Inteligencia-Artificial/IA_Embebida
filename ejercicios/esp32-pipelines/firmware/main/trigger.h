// examples/esp32-cam/firmware/main/trigger.h
#pragma once

typedef enum {
    TRIGGER_SERIAL   = 1,
    TRIGGER_MQTT     = 2,
    TRIGGER_HTTP     = 3,
    TRIGGER_SELFTEST = 4,  /* tecla 't': inferencia sobre las golden images */
} trigger_source_t;

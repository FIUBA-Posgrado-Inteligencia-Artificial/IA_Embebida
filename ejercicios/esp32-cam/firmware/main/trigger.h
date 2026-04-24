// examples/esp32-cam/firmware/main/trigger.h
#pragma once

typedef enum {
    TRIGGER_SERIAL = 1,
    TRIGGER_MQTT   = 2,
} trigger_source_t;

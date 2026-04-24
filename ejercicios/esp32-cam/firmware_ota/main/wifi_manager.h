#pragma once
#include <stdbool.h>
#include "esp_err.h"

esp_err_t wifi_manager_start(void);
bool      wifi_manager_is_connected(void);

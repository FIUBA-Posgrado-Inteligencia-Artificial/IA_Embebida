idf_component_register(
    SRCS "app_main.c"
    INCLUDE_DIRS "." "../.."   # ../.. = wifi_creds.h compartido (examples/esp32-cam/)
    REQUIRES
        nvs_flash
        esp_wifi
        esp_event
        esp_netif
        esp_http_server
        espressif__esp32-camera
)

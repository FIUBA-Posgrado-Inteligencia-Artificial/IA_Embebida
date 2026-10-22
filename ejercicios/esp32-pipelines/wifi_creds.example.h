// Credenciales WiFi compartidas por firmware/ y firmware_test_webserver/.
// Copiar este archivo a wifi_creds.h (gitignoreado) y completar los valores.
// El build falla con "wifi_creds.h: No such file or directory" hasta que
// exista — es intencional.
//
// Password vacía ("") => red abierta (ambos firmwares ajustan el
// threshold.authmode a WIFI_AUTH_OPEN automáticamente).
#pragma once

#define WIFI_SSID     "YourNetwork"
#define WIFI_PASSWORD "YourPassword"

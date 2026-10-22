# TODO — esp32-pipelines

Backlog específico de este subproyecto. El backlog general del curso vive en
`TODO.md` en la raíz del repo.

## Features

- [ ] Script broker MQTT
- [ ] App python simple para MQTT
- [ ] WebApp para MQTT + mostrar cámara
- [ ] Métricas de tiempo de inferencia
- [ ] Métricas de memoria para inferencia
- [ ] Footprint en memoria
- [ ] ¿Entra el firmware?
- [ ] Uso de CPU y memoria
- [x] Script para el build (script05-build.py + main.py menuconfig)
- [x] Agregar QAT a travéss de ONNX?
- [x] Firmware
- [x] Refactor de scripts de entrenamiento -> single script con parámetros
- [x] No se reentrena si ya existe el modelo.
- [x] Entry point
- [x] cambiar cable de puerto USB
- [x] ver si funciona un reset no físico -> NO, no funciona

- [ ] revisar entradas en el entry point
- [ ] On-device validation

### Otros para revisar
- [ ] Dashboard / app GUI.
- [ ] Validación automática on-device (smoke test via MQTT/serial tras flash).
- [ ] Particiones dinámicas según tamaño del modelo.
- [ ] Reporte unificado con diff de métricas (params, MACs, accuracy) antes/después de cada stage.

## ESP32-S3-CAM (placa con PSRAM sospechada; ex example esp32-s3-cam, hoy board `esp32s3cam`)

> La carpeta `examples/esp32-s3-cam/` fue eliminada (2026-08-14); el esp-dl
> parcheado vive ahora vendorizado en
> `firmware/components/esp-dl/`.

### PSRAM de la placa: confirmar el diagnóstico antes de darlo por cerrado
- [x] **Probar el firmware en otra placa ESP32-S3 con PSRAM sana.**
      Hecho 2026-08-13 con la devkit ESP32-S3 N16R8: su PSRAM octal de 8MB
      funciona perfecto con el mismo firmware (good_model corre en 619 ms).
      Refuerza la hipótesis de que la PSRAM de la ESP32-S3-CAM está fallada
      en hardware. (La devkit corre el esp-dl parcheado, así que el
      experimento "otra placa sin parche" sigue abierto abajo.)

  Lo que sí está medido (2026-08-07, ESP32-S3-N16R8 de AliExpress, dos USB-C):
  - Los efuses declaran `PSRAM_CAP=8M`, `PSRAM_VENDOR=AP_3v3`, y GPIO33-37
    alimentados por VDD_SPI. O sea: el chip se presenta como variante con PSRAM.
  - No responde en ninguno de los 4 modos: octal y quad, a 80 y 40 MHz.
    Octal lee ID `0x00000000`, quad lee `0x00ffffff` — los dos patrones de
    "nadie contesta", no de "contesta mal".
  - No es alimentación: la flash externa de 16MB comparte el rail VDD_SPI y
    funciona perfecto. Si el rail estuviera mal, la flash tampoco andaría.
  - Hipótesis abiertas: (a) PSRAM presente en el die pero no funcional
    (falla de fabricación/bonding), (b) chip remarcado — BLOCK1 es programable
    y se puede quemar para que un S3 pelado se presente como R8.

- [ ] Si otra placa funciona sin el parche a esp-dl, revertir el vendorizado
      (ver abajo) y dejar el componente del registry.
- [ ] Si otra placa falla igual, entonces el bug de esp-dl es reproducible en
      hardware sano y vale reportarlo upstream a Espressif.

### esp-dl vendorizado: deuda a revisar en cada actualización
- [ ] `firmware/components/esp-dl/` es una copia
      parcheada. `TensorBase::reset_bias_layout()` (código exclusivo de target S3)
      pide memoria con caps de PSRAM y valida el resultado con un `assert()`, que
      en release se compila a nada: sin PSRAM escribe sobre NULL y el chip panichea.
      El parche agrega el fallback a memoria interna que falta.
- [ ] Al actualizar esp-dl: re-aplicar el parche o verificar que upstream lo arregló.
- [ ] Reportar el bug upstream (el `assert` sin fallback es un problema real
      independientemente de nuestra placa).

# Componentes locales (vendorizados) de este firmware

## `esp-dl/` — copia parcheada de `espressif/esp-dl`

### Qué es

Copia local del componente del registry **`espressif/esp-dl` versión 3.3.9**
(commit upstream `12c0616de145b704e1149c474b9a1e852e631d67`, ver
`esp-dl/idf_component.yml`), sacada de
`../managed_components/espressif__esp-dl` tal como la bajó el component
manager.

### Cómo se activa el override

En `../main/idf_component.yml` la dependencia tiene un `override_path`:

```yaml
espressif/esp-dl:
  version: "^3.3.0"
  override_path: "../components/esp-dl"
```

`override_path` es el mecanismo oficial de ESP-IDF para reemplazar una
dependencia del registry por una copia local: el component manager deja de
descargarla (no vuelve a aparecer en `managed_components/`) y la resuelve
como componente local. Se eligió sobre la alternativa de "poner el
componente en `components/` y confiar en la precedencia" porque deja el
override **explícito y versionado en el manifiesto**, en vez de depender de
una regla de precedencia implícita, y porque evita que el manager baje una
segunda copia y genere un conflicto de nombres.

Cómo verificarlo: después de un build, `dependencies.lock` tiene para
`espressif/esp-dl` un `source: {type: local, path: .../components/esp-dl}`,
`managed_components/` ya no contiene `espressif__esp-dl`, y los paths de
`addr2line` sobre el `.elf` apuntan a `components/esp-dl/...`.

### Qué se parcheó y por qué

Esta placa ESP32-S3 declara 8 MB de PSRAM octal por efuse pero **la PSRAM no
responde en hardware** (falla en los 4 modos octal/quad × 80/40 MHz; ver
`examples/esp32-s3-cam/firmware/sdkconfig.defaults` y el log de
`firmware00-probe`). El firmware arranca igual gracias a
`CONFIG_SPIRAM_IGNORE_NOTFOUND=y`, pero esp-dl **hardcodea
`MALLOC_CAP_SPIRAM`** en lugares que no necesitan RAM externa:

- `dl/module/include/dl_module_base.hpp` — `Module::operator new` →
  `malloc_aligned(size, MALLOC_CAP_SPIRAM)`
- `dl/tensor/include/dl_tensor_base.hpp` — ídem para `TensorBase`
- `dl/tensor/src/dl_tensor_base.cpp` — `TensorBase::reset_bias_layout()`,
  el repaqueteo de bias al formato QACC específico de S3

Sin PSRAM esa capability tiene 0 bytes y todas esas alocaciones devuelven
`NULL`. El que crashea primero en la práctica es `Module::operator new`
(88 bytes, `sizeof(Conv)`): es el primer `new` que ejecuta `Model::load()`. `operator new` no está declarado `noexcept`, así que el compilador
**no** chequea el resultado: el constructor corre con `this == NULL` y
escribe el puntero a vtable en la dirección 0 → `Guru Meditation Error
(StoreProhibited)`, `EXCVADDR: 0x00000000`, boot loop determinístico al
cargar el modelo.

Los parches aplicados:

1. **`dl/tool/src/dl_tool.cpp`** (el que resuelve el problema): si un
   `malloc_aligned()` / `calloc_aligned()` que pidió `MALLOC_CAP_SPIRAM`
   falla, se reintenta una vez con `MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT`
   antes de rendirse, con un `ESP_LOGW` una sola vez. Si el reintento
   también falla, se devuelve `NULL` igual que upstream, así que los callers
   que sí chequean no cambian de comportamiento. Es un solo punto central en
   vez de parchear cada sitio que hardcodea PSRAM.
2. **`dl/tensor/src/dl_tensor_base.cpp`**, `reset_bias_layout()` rama
   `QUANT_TYPE_SYMM_8BIT`: ese `calloc_aligned()` estaba protegido sólo por
   un `assert()`, que en build release (`NDEBUG`) es no-op, y el loop de
   empaquetado escribía sobre el `NULL`. Ahora hay fallback explícito a RAM
   interna y, si tampoco alcanza, un `ESP_LOGE` con el tamaño pedido y la
   memoria libre + `abort()` controlado. Es endurecimiento: con el parche 1
   esta ruta ya no debería fallar, pero deja de ser una corrupción silenciosa
   si alguna vez hay OOM real.

   La rama `QUANT_TYPE_SYMM_16BIT` de la misma función tiene el mismo patrón
   (upstream ni siquiera tiene un `assert()` ahí) y recibió el mismo
   endurecimiento.

3. **`dl/module/include/dl_module_base.hpp`** y
   **`dl/tensor/include/dl_tensor_base.hpp`**, los dos `operator new`: si
   `malloc_aligned()` devuelve `NULL` —lo que ahora sólo puede pasar si
   *también* falló el fallback del parche 1, es decir OOM real— se loguea un
   `ESP_LOGE` con el tamaño pedido y la memoria libre de ambas capabilities y
   se hace `abort()`. Sin esto, un OOM real vuelve al mismo
   `StoreProhibited` en la dirección 0, indiagnosticable. Se prefirió
   log + `abort()` antes que declararlos `noexcept` (que forzaría el
   null-check del compilador pero movería el crash a un punto posterior y más
   difícil de leer).

Los comentarios dentro de `esp-dl/` están **en inglés**, a diferencia del
resto del repo, para ser consistentes con el código que los rodea y para que
un eventual reporte/PR río arriba a Espressif se pueda copiar tal cual.

### Cómo reencontrar el parche

Todas las divergencias contra upstream están marcadas con el prefijo
`PATCH:`:

```bash
grep -rn "PATCH:" examples/esp32-s3-cam/firmware/components/esp-dl/
```

### Otra divergencia: `fbs_loader/lib/` podado

Upstream trae librerías precompiladas para los 9 targets soportados (~40 MB).
Se dejó **sólo `fbs_loader/lib/esp32s3/`** para no meter 36 MB de binarios
inútiles en el repo. Este firmware es exclusivamente ESP32-S3; si alguien
cambia el target va a fallar el link con un "no se encuentra libfbs_model.a"
y tiene que recuperar el directorio del registry.

### Qué hacer al actualizar esp-dl

1. Sacar temporalmente el `override_path` de `../main/idf_component.yml`,
   subir el rango de versión y buildear para que el manager baje la versión
   nueva a `managed_components/espressif__esp-dl`.
2. Revisar si los bugs de arriba siguen presentes en la versión nueva
   (buscar `MALLOC_CAP_SPIRAM` en `dl_module_base.hpp`, `dl_tensor_base.hpp`
   y `dl_tool.cpp`). Si Espressif los arregló, **borrar este directorio y el
   `override_path`**: el vendorizado deja de tener razón de ser.

   Si la actualización incluye subir de ESP-IDF, resolver además el
   `PATCH-NOTE (deferred)` de `dl/tool/src/dl_tool.cpp` (`retry_from_internal`,
   ~línea 223): el reintento a RAM interna **descarta `MALLOC_CAP_SIMD`**, que
   el camino primario (`malloc_aligned`/`calloc_aligned`, ~línea 238) agrega
   bajo `SOC_SIMD_INSTRUCTION_SUPPORTED`. Es inocuo en IDF 5.3.2 porque esa
   rama no compila ahí, pero **muerde en IDF ≥ 5.5**: los tensores que caen al
   fallback quedarían sin la capability que los kernels SIMD asumen. Al pasar a
   IDF ≥ 5.5 hay que propagar `MALLOC_CAP_SIMD` al reintento (o verificar que
   el allocator lo garantice) antes de dar el parche por bueno.
3. Si siguen, reemplazar el contenido de `esp-dl/` por la copia nueva,
   volver a podar `fbs_loader/lib/`, reaplicar los bloques `PATCH:` (usar el
   `grep` de arriba sobre la copia vieja como checklist), borrar
   `.component_hash` **y `CHECKSUMS.json`** (son artefactos del registry que
   dejan de ser válidos apenas se toca un archivo), y reponer el
   `override_path`.
4. Verificar en hardware: el firmware tiene que arrancar, loguear
   `ESP-DL model loaded` y correr una inferencia disparada por serie sin
   panichear.

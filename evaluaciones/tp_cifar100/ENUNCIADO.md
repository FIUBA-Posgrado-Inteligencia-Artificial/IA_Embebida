# TP final: competencia de compresión

**IA Embebida — CEIA / CESE — edición a25**

## Resumen

Les damos un modelo que clasifica imágenes de CIFAR-100 y que es demasiado grande y lento para un microcontrolador. Su trabajo es **comprimirlo** con las técnicas de la materia (cuantización, pruning, factorización, NAS, knowledge distillation, optimización algorítmica) hasta que entre en un **ESP32-S3** cumpliendo los requisitos de la aplicación.

Es una **competencia**: entre los modelos que cumplen los requisitos, gana el más comprimido. El ganador se lleva un **10**.

No hace falta tener la placa. Todo se mide en la PC con **int8 simulado** (esp-ppq, el cuantizador oficial de Espressif para ESP-DL). El modelo ganador lo desplegamos en vivo en un ESP32-S3 el día de la presentación.

## La aplicación

Un clasificador de objetos que corre en un **ESP32-S3 sin PSRAM**: sólo tiene la SRAM interna (512 KB, de los que el resto del firmware deja libres unos 300 KB) y la flash. El producto necesita:

- **Calidad**: reconocer las 100 clases de CIFAR-100 con una accuracy mínima.
- **Flash**: el modelo comparte la flash con el firmware y con las actualizaciones OTA, así que su tamaño importa.
- **RAM**: las activaciones intermedias viven en la SRAM interna, compartida con la cámara, el WiFi y el resto de la aplicación.
- **Tiempo real**: cuantas menos operaciones, más cuadros por segundo y menos energía por inferencia.

## El modelo provisto (baseline)

Una ResNet-56 para CIFAR (He et al., 2016), entrenada por nosotros. Está en el repositorio de la materia, en `evaluaciones/tp_cifar100/`, como `baseline_resnet56.pth` (PyTorch) y `baseline_resnet56.onnx`.

| Métrica | Baseline |
|---|---:|
| Parámetros | 859.492 |
| Pesos (archivo `.espdl` int8) | 949 KB |
| MACs por inferencia | 125,8 M |
| Pico de activaciones | 48 KB |
| Accuracy top-1 float | 72,1 % |
| Accuracy top-1 int8 simulada | 71,5 % |

**El baseline no cumple ninguno de los requisitos de memoria ni de cómputo.** Pasarlo a int8 no alcanza: los 949 KB ya son int8. Hay que aplicar técnicas de compresión de verdad.

Pueden usar el baseline como punto de partida (podarlo, factorizarlo, cuantizarlo) o como *teacher* para knowledge distillation, o diseñar una arquitectura nueva. Vale todo lo que respete las reglas.

## Requisitos y niveles

Los niveles son **independientes**: cada uno es un punto distinto del compromiso entre tamaño y calidad (la frontera de Pareto). Un nivel se alcanza cumpliendo **todas las columnas de su fila**; no hace falta cumplir los anteriores. Por ejemplo, un modelo de 60 KB, 1,5 M de MACs, 12 KB de pico y 53 % es Oro, aunque no llegue a la accuracy de Bronce.

| Nivel | Pesos (`.espdl`) | Pico de activaciones | MACs | Accuracy int8 |
|---|---:|---:|---:|---:|
| Bronce | ≤ 160 KB | ≤ 32 KB | ≤ 30 M | ≥ 63 % |
| Plata | ≤ 128 KB | ≤ 24 KB | ≤ 8 M | ≥ 62 % |
| Oro | ≤ 64 KB | ≤ 16 KB | ≤ 2 M | ≥ 52 % |

Además, en todos los niveles:

- La entrada es fija: un tensor float `[1, 3, 32, 32]`, RGB, normalizado con la media y el desvío de `datos.py`. La salida son los 100 logits.
- Todo el modelo se cuantiza a int8 y sólo usa operaciones soportadas por ESP-DL (la lista está en `OPS_PERMITIDOS` de `evaluar.py`; si necesitan otra, pregunten).

Existe al menos un modelo de referencia, entrenado por nosotros sin optimizarlo demasiado, que alcanza cada nivel. Las accuracies mínimas se fijaron unos puntos por debajo de esas referencias.

### Qué mide cada requisito

- **Pesos**: el tamaño en bytes del archivo `.espdl` que genera esp-ppq. Es exactamente lo que va a la flash. Ojo: incluye unos ~24 KB de metadatos y bias además de 1 byte por peso, así que también cuenta tener menos capas.
- **Pico de activaciones**: la RAM mínima para las activaciones. Se recorre el grafo capa por capa y, en cada capa, se suman los bytes (int8, 1 byte por valor) de todos los tensores que todavía hacen falta: la entrada de la capa, su salida y cualquier tensor que espera para usarse más adelante (por ejemplo, la rama residual de una ResNet). Las activaciones (ReLU, etc.), BatchNorm y los reshapes no ocupan un buffer nuevo. El pico es el máximo sobre todas las capas.
    - Ejemplo: un bloque residual con tensores de 32×32×16 tiene vivos a la vez la entrada (que espera para la suma), la salida de la primera conv y la de la segunda: 3 × 16 KB = 48 KB.
- **MACs**: multiplicaciones-acumulaciones de las capas Conv, Gemm y MatMul para una imagen.
- **Accuracy int8**: top-1 sobre las 10 000 imágenes de test de CIFAR-100, con el modelo cuantizado a int8 por esp-ppq (configuración fija, calibración con 512 imágenes de val) y ejecutado en el simulador de esp-ppq.

## La competencia

Compiten por el 10 las entregas que **cumplen la fila de Bronce**, incluida su accuracy mínima (≥ 63 %). La idea es comprimir lo más posible **sin resignar calidad**: un modelo Oro con menos accuracy aprueba con buena nota, pero no compite.

Entre las que compiten, gana el **menor score**:

```
score = ½ · pesos / pesos_baseline  +  ½ · MACs / MACs_baseline
```

Con pesos y MACs medidos como arriba. El baseline tiene score 1. Si hay empate a 4 decimales, gana la mayor accuracy int8, y si sigue el empate, la entrega más temprana.

Además de la competencia hay **menciones** (sin nota extra): mayor accuracy dentro del nivel Oro y menor pico de activaciones con nivel Plata o superior.

## Cómo se evalúa

Las métricas las calcula siempre `evaluar.py`, nunca se reportan a mano. Lo pueden correr ustedes todas las veces que quieran:

```bash
cd evaluaciones/tp_cifar100
uv sync
uv run python evaluar.py mi_modelo.onnx          # evaluación completa (minutos; más rápido con GPU)
uv run python evaluar.py mi_modelo.onnx --rapido # sólo MACs, pico y ops (segundos)
```

Para exportar el modelo desde PyTorch con la interfaz correcta:

```python
from modelos import exportar_onnx
exportar_onnx(mi_modelo, "mi_modelo.onnx")   # entrada fija [1, 3, 32, 32], opset 13
```

El script también deja el `.espdl`, que es el archivo que se flashea en la placa.

**La cuantización la hace `evaluar.py` con una configuración fija.** Si quieren que su modelo sobreviva mejor a la cuantización, el trabajo lo tienen que hacer antes: entrenamiento consciente de la cuantización (QAT), regularización, rangos de activación acotados (ReLU6), etc. Lo que entregan es el modelo float y el evaluador lo cuantiza.

## Reglas

1. **El test de CIFAR-100 no se usa para entrenar ni para elegir modelos.** Usen el split que arma `datos.py`: 45 000 imágenes de train y 5 000 de val, con semilla fija. El test es sólo para la evaluación final.
2. **Nada de datos externos ni de pesos preentrenados externos** (ImageNet, etc.). Sólo CIFAR-100 train/val y el baseline provisto. Las técnicas de data augmentation están permitidas.
3. **Reproducibilidad**: el código de entrenamiento tiene que estar en la entrega. A los tres primeros puestos les vamos a reentrenar el modelo con su código; si el resultado no se reproduce (dentro de un margen razonable), pierden el puesto.
4. **Ensembles y test-time augmentation**: no. Una imagen, una pasada del modelo.
5. Grupos de hasta 3 personas (también se puede individual).

## Entregas y fechas

| Fecha | Qué |
|---|---|
| Jueves 19/11 | Publicación del enunciado y del baseline |
| Jueves 26/11 y 3/12 | Entregas intermedias opcionales: suben su `.onnx` y publicamos un **leaderboard** |
| Miércoles 9/12, 23:59 | Entrega final |
| Jueves 10/12 (clase 8) | Presentaciones, defensa y demo en vivo del ganador |

La entrega final es un repositorio (o un `.zip`) con:

- `modelo.onnx`: el modelo que compite.
- El **código** completo para reproducirlo: entrenamiento, compresión y export.
- Un **informe breve** (puede ser el `README.md`) que cuente el camino: qué técnicas probaron, en qué orden, qué funcionó, qué no y por qué, con los resultados de `evaluar.py` de cada paso. Un gráfico de accuracy contra score (su propia frontera de Pareto) suma mucho.

## Presentación y demo en vivo

Cada grupo presenta en ~10 minutos el camino que siguió. **No hace falta tener una placa ni desplegar.**

Al **ganador** le pedimos que muestre en vivo la cadena completa:

1. Correr `evaluar.py` sobre su entrega y mostrar el resultado.
2. Explicar cómo llegó al modelo (sin reentrenar en vivo).
3. Nosotros tomamos su `.espdl`, lo flasheamos en un ESP32-S3 y lo corremos en la placa.

Si no tienen placa, el paso 3 lo hacemos nosotros. La accuracy en la placa es **informativa**: el simulador de esp-ppq y los kernels de ESP-DL redondean distinto y pueden diferir en algún punto. El resultado oficial es el de `evaluar.py`.

## Nota

| Resultado | Nota |
|---|---|
| Ganador de la competencia | **10** |
| Oro | 8 a 9 |
| Plata | 7 a 8 |
| Bronce | 6 a 7 |
| No alcanza ningún nivel | No aprueba con este enunciado (se puede reentregar) |

Cuenta el nivel más alto alcanzado. Dentro de cada rango decide la calidad del informe y de la defensa: que expliquen qué hicieron y por qué, no sólo el número final. El 2.º y el 3.º puesto de la competencia suman 1 punto (con máximo 10).

## Alternativa: propuesta propia

Como dijimos en la clase 1, quien prefiera trabajar con un problema propio (por ejemplo, relacionado con su trabajo final) puede proponerlo en lugar de este enunciado. Tiene que incluir compresión de un modelo y, si es posible, un despliegue o un benchmark en hardware real. Esas entregas no compiten: se evalúan con el informe y la defensa. Escríbannos antes del **jueves 26/11** con la propuesta.

## Pistas

- Mirar dónde están los MACs y el pico de activaciones (`evaluar.py --rapido` dice en qué capa ocurre el pico). En CIFAR, las primeras capas a 32×32 suelen dominar las dos cosas.
- El clasificador final (`Linear` de 100 salidas) es un costo fijo de pesos que no se puede evitar del todo.
- Knowledge distillation desde el baseline ayuda, pero no es gratis: hay que ajustar la temperatura y el peso de la pérdida de KD. En nuestras pruebas, con los mismos hiperparámetros, sumó ~1,4 puntos a un modelo mediano y le restó ~3 a uno muy chico.
- La cuantización int8 no es inocua: en algunas de nuestras referencias costó menos de 1 punto y en otras 6. Si su modelo pierde mucho, prueben QAT o acotar las activaciones.
- Pruning estructurado (canales enteros) reduce pesos, MACs y activaciones. Pruning no estructurado no reduce nada de lo que mide el evaluador: el `.espdl` guarda los ceros igual.
- Conv → BatchNorm → ReLU permite plegar el BatchNorm en la convolución (lo hace esp-ppq solo).
- Primero logren algún nivel y después optimicen. Una entrega válida temprana en el leaderboard vale más que una perfecta que no llega.

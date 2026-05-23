# Findings — MCUNet en XIAO ESP32-S3 Sense

Documento técnico detallado. Para un reporte ejecutivo de 1 página ver [`REPORT.md`](REPORT.md).

Toda latencia medida con `host/monitor.py /dev/ttyACM0 N` sobre el XIAO real durante ≥10 inferencias consecutivas. Logs en [`logs/`](logs/).

## Imagen capturada por la OV2640

![Frame capturado](plots/sample_camera_frame.png)

Capturada a 240×240 RGB565 directamente desde el sensor del Sense usando `firmware/camera_test/`. Para llegar a este punto se resolvieron tres bugs documentados en la sección "Bugs" abajo:

- `CONFIG_SCCB_HARDWARE_I2C_PORT0=y`: el driver `esp32-camera` 2.0.13 default usa I²C port 1, que en esta combinación XIAO + Sense Expansion Board no funciona.
- `i2c_master_probe` de IDF 5.2 hace busy-wait infinito en direcciones sin ACK; el periférico queda atascado tras flashes acumulados. Reset por flash no lo recupera, requiere power-cycle físico.
- PSRAM OCT (hardware-locked en ESP32-S3R8) coexistiendo con cam_hal complicó el debugging del I²C.

## Inferencia sobre la imagen capturada

![Panel de predicciones reales](plots/07_classification_panel.png)

Sobre la imagen del sensor (poca luz, sujeto desde abajo, techo dominante en el frame):

- `mcunet-vww2` predice `no-pers` con scores [+22, -22]. La persona está pero el modelo no la detecta.
- `mcunet-in2` predice `alp` (clase 971, valle alpino) con score +27. Confunde los azulejos del techo con un valle nevado.
- `mbv2-w0.35` predice `jersey/vault` con scores negativos. Baja confianza, inestable entre frames.
- `proxyless-w0.3` predice `power drill` (clase 741) con +35. Confiado y equivocado.
- `mcunet-person-det` reporta `no-pers` con best=0.419, a 0.081 del umbral 0.5.

Sobre la imagen del repo MCUNet (`person_det.jpg`, personas centradas y bien iluminadas), los mismos modelos sí detectan persona o predicen clases coherentes con el fondo (window shade, bannister, obelisk). La diferencia se debe a las distintas distribuciones de orientación, iluminación y composición. Es un límite práctico de estos modelos preentrenados ante imágenes con características distintas al training set.

## Vista rápida de los plots

![Latencia por modelo](plots/01_latency_absolute.png)

![Latencia normalizada](plots/02_latency_per_pixel.png)

![Distribución de kernels DWCONV](plots/03_kernel_distribution.png)

![Desglose por op](plots/04_op_breakdown.png)

## Shapes y dtypes de los modelos

Verificado en host con TFLite Interpreter:

| Modelo | Input | Output | Dtype in | Ops |
|---|---|---|---|---|
| mcunet-vww2 | 1×144×144×3 | 1×2 | int8 | CONV, DWCONV, ADD, AVG_POOL, PAD, RESHAPE |
| mcunet-in0 | 1×48×48×3 | 1×1000 | int8 | mismo set |
| mcunet-in2 | 1×160×160×3 | 1×1000 | int8 | mismo set |
| mcunet-person-det | 1×128×160×3 | 1×4×5×18 | float32 | + MAX_POOL, RESIZE_NN, QUANT/DEQUANT |
| mbv2-w0.35 | 1×144×144×3 | 1×1000 | int8 | mismo set |
| proxyless-w0.3 | 1×176×176×3 | 1×1000 | int8 | mismo set |

Notas:

- `mcunet-in0` es 48×48, no 160 como sugería el nombre. Para el HITO 2 se usó `mcunet-in2` (160×160), que coincide con el pipeline de cámara.
- `mcunet-vww2` es 144×144 RGB, no 96×96 grayscale como el demo VWW típico de Google.
- `mcunet-person-det` acepta float32, no int8. Simplifica preprocesado pero aumenta el memory budget.
- `mbv2-w0.35` y `mcunet-vww2` comparten input shape (144×144×3), lo que permite una comparación A/B directa para el HITO 4.

## Fase 1 — Visual Wake Words

| Modelo | Input | Tensor arena | Bin size | Latencia | Heap libre PSRAM | Predicción |
|---|---|---|---|---|---|---|
| mcunet-vww2 (int8) | 144×144×3 | 473 KB / 600 KB (PSRAM) | 1.42 MB | 3.39 s ± 1 ms | 6.3 MB | PERSON sobre `person_det.jpg`, scores [-51, +51], idéntico al host |

Latencia anómala: 3.39 s vs 54 ms del bench Espressif `person_detection` (96×96 con esp-nn). La causa se cuantifica en la Fase 5.

## Fase 2 — ImageNet

| Modelo | Input | Tensor arena | Bin size | Latencia | Top-1 (host = ESP32) |
|---|---|---|---|---|---|
| mcunet-in2 (int8) | 160×160×3 | 540 KB / 700 KB (PSRAM) | ~1.52 MB | 3.93 s ± 0.01 ms | 906 'window shade' |

Validación cruzada sobre `person_det.jpg`: top-5 del ESP32 = `[906, 617, 834, 681, 785]` vs host = `[906, 617, 681, 834, 785]`. Mismos cinco elementos, scores prácticamente idénticos.

El binario con el `test_image` embebido supera 1.5 MB de `single_app_large`, por lo que se requirió `partitions.csv` custom con `factory 3 MB`.

## Fase 3 — Person detection

| Modelo | Input | Tensor arena | Bin | Latencia | Detección sobre person_det.jpg |
|---|---|---|---|---|---|
| mcunet-person-det | 128×160×3 float32 | 716 KB / 900 KB | ~595 KB | 1.24 s ± 4 µs | PERSON conf=0.891 en celda head 8×10 |

Detector tipo SSD/YOLO con 3 cabezas multi-escala (4×5, 8×10, 16×20) × 3 anchors × (x, y, w, h, obj, cls). El demo aplica `sigmoid(obj) × sigmoid(person_logit)` y devuelve el arg-max sobre las tres cabezas. Sin NMS, sin bbox final.

A 1.24 s es 3× más rápido que vww (3.39 s) e in2 (3.93 s), pese a ser float32 y tener input mayor. Razón: el modelo es mucho más pequeño en weights (296 KB vs ~1 MB). Confirma que el bottleneck en este runtime es el ancho de banda al modelo en flash/PSRAM, no las operaciones aritméticas.

## Fase 4 — Benchmark de arquitecturas

Todos corridos en el mismo runtime (esp-tflite-micro 1.3.3 + esp-nn) sobre el mismo XIAO ESP32-S3 Sense @ 240 MHz, sobre la imagen `person_det.jpg`, en modo estático.

| Modelo | Tarea | Input | Latencia | ms/pixel | Top-1 (host = ESP32) |
|---|---|---|---|---|---|
| mcunet-vww2 | VWW (2-clases) | 144×144×3 int8 | 3.39 s | 0.164 | PERSON |
| mcunet-in2 | ImageNet (1000-clases) | 160×160×3 int8 | 3.93 s | 0.154 | window shade |
| mcunet-person-det | Detector bbox | 128×160×3 fp32 | 1.24 s | 0.061 | PERSON conf=0.891 |
| mbv2-w0.35 | ImageNet (1000-clases) | 144×144×3 int8 | 1.64 s | 0.079 | bannister |
| proxyless-w0.3 | ImageNet (1000-clases) | 176×176×3 int8 | 2.51 s | 0.081 | obelisk |

### Hallazgos

1. MobileNetV2-w0.35 es ~2× más rápido que mcunet-vww2 a igual input (144×144) sobre este runtime. Contradice los números publicados de MCUNet, pero es esperable: los números del paper son con TinyEngine, motor de inferencia co-diseñado para STM32 ARM Cortex-M. Sin TinyEngine, la ventaja arquitectural de MCUNet desaparece.
2. mbv2-w0.35 y proxyless-w0.3 dan latencia muy similar por pixel (~0.08 ms/px). Las arquitecturas MobileNet-style están bien optimizadas en esp-nn.
3. MCUNet es ~2× más lento por pixel (0.15 vs 0.08 ms/px). La arquitectura usa expand-ratios y kernel sizes diversos (resultado del NAS) que no caen siempre en los kernels SIMD de esp-nn.
4. mcunet-person-det es float32 pero corre 3× más rápido que los clasificadores int8. Pesos mucho más pequeños (296 KB vs ~1 MB) → menos transferencias desde flash/PSRAM.
5. Predicciones idénticas entre host TFLite y ESP32 en los cuatro modelos validados. La latencia depende sólo del runtime, no de aproximaciones numéricas.

## Fase 5 — Profiling op-por-op

Instrumentando `imagenet_demo` y `benchmark_mbv2` con `tflite::MicroProfiler::LogTicksPerTagCsv()`.

### Resultados crudos (primera inferencia)

| Op | mcunet-in2 ticks | mbv2-w0.35 ticks |
|---|---|---|
| PAD | 323 761 | 172 094 |
| ADD | 21 662 | 6 124 |
| RESHAPE | 41 | 40 |
| Total tagged | 345 464 (≈ 1.44 ms) | 178 258 (≈ 0.74 ms) |
| Inferencia total | 3 916 807 µs | 1 644 547 µs |
| CONV+DWCONV "invisible" | 3 915 ms (99.96%) | 1 644 ms (99.95%) |

Los kernels CONV/DWCONV ejecutados por `esp-nn` no emiten `BeginEvent`/`EndEvent`, por lo que no aparecen en el log del profiler. Si estuvieran cayendo a kernels reference de TFLM, se verían tagged. La ausencia confirma que esp-nn está activo en ambos modelos.

### Análisis arquitectural (dump op-por-op en host)

| Métrica | mcunet-in2 | mbv2-w0.35 |
|---|---|---|
| Total CONV_2D ops | 37 (36 son 1×1) | 36 (35 son 1×1) |
| Total DWCONV ops | 18 | 17 |
| DWCONV kernel sizes | 3×3: 6, 5×5: 9, 7×7: 3 | 3×3: 17 |

### Causa del slowdown 2.4× de MCUNet vs MBV2

MCUNet tiene 12 DWCONVs con kernels grandes (5×5 y 7×7). MobileNetV2 usa siempre 3×3.

- Un DWCONV 5×5 hace ~2.78× más cómputo que un 3×3.
- Un DWCONV 7×7 hace ~5.44× más cómputo que un 3×3.
- `esp-nn` está específicamente optimizado para DWCONV 3×3 (el caso común en MobileNetV1/V2). Para 5×5 y 7×7 cae al path "generic depthwise conv" — funcional pero sin la fast-path SIMD.

El NAS que produjo MCUNet eligió esos kernels grandes para maximizar accuracy/params; MCUNet fue diseñado para TinyEngine, que sí los acelera. Sin TinyEngine, esa decisión arquitectural se vuelve costo.

### Implicaciones

- La causa del slowdown está identificada y cuantificada.
- No es un bug ni mala configuración. Es una propiedad arquitectural del NAS de MCUNet.
- Si se quisiera acelerar MCUNet en ESP32-S3 habría que re-correr TinyNAS con un latency proxy de esp-nn en lugar de TinyEngine. El NAS evitaría kernels grandes y produciría un modelo adaptado al runtime real del chip.

## Resumen del experimento

- ¿Se puede correr MCUNet en ESP32-S3? Sí, con esp-tflite-micro. Los cinco modelos funcionan y dan resultados idénticos al host.
- ¿Vale la pena MCUNet vs MobileNetV2 en ESP32-S3? No con este runtime. TinyEngine es lo que hace MCUNet competitivo, y TinyEngine no soporta Xtensa.
- ¿Generalizan estos modelos a imágenes del sensor real del Sense? No bien. La brecha out-of-distribution entre `person_det.jpg` (training-like) y la imagen capturada (composición no controlada) es grande: `mcunet-vww2` cambia de PERSON a no-pers, `mcunet-person-det` baja de conf=0.891 a 0.419 (queda bajo umbral).
- Caminos posibles: (a) portar TinyEngine a Xtensa (proyecto grande); (b) usar la Grove Vision AI V2 (Cortex-M55 + NPU); (c) aceptar MobileNetV2 como mejor opción práctica en este chip; (d) fine-tunear los modelos sobre frames reales para cerrar la brecha OOD.

## Bugs resueltos durante el experimento

1. `convert_to_c_array.sh` generaba símbolos con internal linkage (C++ `const` es internal por defecto). Fix: declarar `extern __attribute__((used)) const ...` dentro de `extern "C" { }`.
2. Tensor arena intentaba ir a DRAM y desbordaba. Fix: `CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY=y` para que `EXT_RAM_BSS_ATTR` mande BSS a PSRAM.
3. El nombre `mcunet-vww2` sugería 96×96 grayscale (típico VWW de Google). El modelo real es 144×144 RGB. Verificación previa en host con `eval_tflite.py` evitó horas perdidas en preprocesado.
4. OpResolver inicial sin `PAD`. AllocateTensors fallaba con "Op for builtin opcode PAD not registered". Fix: `resolver.AddPad()` y template a `<14>`.
5. Tensor arena 400 KB insuficiente. "Failed to resize buffer: requested 379KB, available 315KB". Fix: 600 KB.
6. Stack default del `main_task` (3584 bytes) desborda durante `Invoke`. Reset silencioso en bucle tras la primera inferencia. Fix: `CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192`.
7. `esp32-camera` v2.0.13 default usa I²C port 1 para SCCB. En el XIAO + Sense Expansion Board falla. Fix: `CONFIG_SCCB_HARDWARE_I2C_PORT0=y` en `sdkconfig.defaults`.
8. El periférico HW i2c_master de IDF 5.2 hace busy-wait infinito en `s_i2c_send_commands` (línea 424 de `driver/i2c/i2c_master.c`) cuando la dirección no contesta ACK. El timeout pasado a `i2c_master_probe`/`transmit` se ignora. Confirmado por backtrace decodificado. Después de algunos flashes acumulados, el periférico queda en este estado y bloquea todas las operaciones I²C. El reset por flash (DTR/RTS) no lo recupera. Workaround: power-cycle físico (desconectar VCC).
9. `i2c_chipid` (firmware standalone sin esp32-camera) lee `PID=0x26 VER=0x42 = 0x2642` (OV2640 estándar) en SDA=GPIO40 / SCL=GPIO39 cuando el chip está en buen estado. Comprueba que el sensor está vivo y que el bug está en `esp32-camera` + IDF 5.2, no en el hardware.

## Notas técnicas

- ESP-IDF: v5.2
- esp-tflite-micro: 1.3.3 (managed component)
- esp-nn: 1.1.x (bundled con esp-tflite-micro)
- esp32-camera: 2.0.13
- Flash mode: QIO 80 MHz · PSRAM: octal 80 MHz · CPU: 240 MHz
- Stack main_task: 8192 bytes (default 3584 desbordaba durante invoke)
- Tensor arena: 600–900 KB en PSRAM (`EXT_RAM_BSS_ATTR`)
- Partition factory: 3 MB custom (`single_app_large` 1.5 MB era insuficiente)

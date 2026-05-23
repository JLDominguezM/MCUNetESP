# MCUNetESP

Despliegue de modelos [MCUNet (MIT Han Lab)](https://github.com/mit-han-lab/mcunet) en una **XIAO ESP32-S3 Sense** usando `esp-tflite-micro` + `esp-nn`.

- Reporte ejecutivo: [`docs/REPORT.md`](docs/REPORT.md)
- Tabla técnica completa: [`docs/findings.md`](docs/findings.md)

## Brecha out-of-distribution

![Ideal vs real](docs/plots/09_comparison_ideal_vs_real.png)

El mismo `mcunet-vww2` predice `PERSON` con score +51 sobre la imagen del repo MCUNet, y `no-pers` con score +22 sobre una foto tomada por la propia cámara del Sense. El detector `mcunet-person-det` baja de `conf=0.891` a `conf=0.419` (umbral 0.5). Los clasificadores ImageNet predicen objetos diferentes pero igualmente desconectados del sujeto (no existe "persona" como clase en ImageNet). El experimento mide una brecha de generalización medible sin re-entrenamiento sobre la distribución real de la aplicación.

## Estado

Cinco hitos completados:

| # | Hito | Resultado |
|---|---|---|
| 1 | VWW (mcunet-vww2) | PERSON detectado, 3.39 s |
| 2 | ImageNet (mcunet-in2) | top-5 idéntico host/ESP32 |
| 3 | Person detection (bbox) | PERSON conf=0.891 en celda 8×10 |
| 4 | Benchmark vs MBV2 + ProxylessNAS | MCUNet 2.4× más lento |
| 5 | Profiling op-por-op | causa identificada: DWCONV 5×5/7×7 |

## Resultados

### Latencia por inferencia

![Latencia absoluta](docs/plots/01_latency_absolute.png)

### Latencia normalizada por pixel

![Latencia por pixel](docs/plots/02_latency_per_pixel.png)

### Distribución de kernels DWCONV

![Distribución de kernels](docs/plots/03_kernel_distribution.png)

`esp-nn` está optimizado para DWCONV 3×3. MCUNet (NAS-generated) usa 12 DWCONVs con kernels 5×5 y 7×7 — diseñados para correr en TinyEngine, no en esp-nn. Es la causa del slowdown 2.4×.

## Pipeline

![Pipeline de inferencia](docs/plots/08_pipeline.png)

## Imagen capturada por la OV2640

![Frame real](docs/plots/sample_camera_frame.png)

Capturada a 240×240 RGB565 directamente por el sensor del Sense. Para destrabar la cámara hizo falta forzar `CONFIG_SCCB_HARDWARE_I2C_PORT0=y` en `sdkconfig.defaults` y reiniciar el chip por power-cycle físico tras varios flashes acumulados. Detalles en [`docs/findings.md`](docs/findings.md).

## Demo en vivo (vww_demo)

Una vez la cámara funciona y el firmware `vww_demo` se reflashea con el Kconfig SCCB correcto, el ciclo es: capturar frame → preprocesar → invocar modelo → log. Sin imagen estática embebida, el modelo predice sobre lo que ve en cada iteración.

Log real de 10 inferencias consecutivas apuntando la cámara a una persona en condiciones de iluminación ambiental (`docs/logs/vww_demo_live.log`):

```
I camera: Camera PID=0x26 VER=0x42 MIDL=0x7f MIDH=0xa2
I camera: Detected OV2640 camera at address=0x30
I vww: model OK: in=[1,144,144,3] type=9  out=[1,2] type=9
I vww: arena used: 473644 / 614400 bytes
I vww: no-pers  scores=[   1,  -1]  inv=3.40 s
I vww: PERSON   scores=[ -11,  11]
I vww: no-pers  scores=[  13, -13]
I vww: PERSON   scores=[ -11,  11]
I vww: PERSON   scores=[  -3,   3]
I vww: no-pers  scores=[   0,   0]
I vww: no-pers  scores=[   3,  -3]
I vww: PERSON   scores=[  -2,   2]
I vww: no-pers  scores=[   4,  -4]
I vww: no-pers  scores=[  12, -12]
```

En 10 inferencias seguidas, 4 dieron PERSON y 6 no-pers, con scores absolutos bajos (mayoría entre ±5). Esto confirma cuantitativamente la brecha out-of-distribution documentada arriba: el modelo está al borde de la decisión, oscilando entre detección y no-detección frame a frame en condiciones reales no controladas. Sobre la imagen del repo MCUNet (entrenamiento ideal), los mismos modelo y firmware dan scores ±51 (decisión clara).

La imagen se redimensiona al input de cada modelo:

![Imagen redimensionada por input size](docs/plots/05_test_image_resized.png)

## Detector multi-escala

`mcunet-person-det` tiene 3 cabezas a strides distintos. La celda con mayor confianza queda marcada:

![Grids de detección](docs/plots/06_person_det_grid.png)

Sobre la imagen del repo MCUNet, la cabeza media (stride 16) detecta persona con confianza 0.891. Las otras dos dieron 0.320 y 0.051.

## Top-5 por modelo

![Panel de clasificación](docs/plots/07_classification_panel.png)

Los números corresponden a las inferencias del propio XIAO ESP32-S3. Coinciden bit-a-bit con el host TFLite (validado con `host/reproduce_host_inference.py`).

## Tabla de resultados

| Modelo | Tarea | Latencia | Predicción (imagen del repo) |
|---|---|---|---|
| mcunet-vww2 | VWW (persona/no) | 3.39 s | PERSON |
| mcunet-in2 | ImageNet 1000 | 3.93 s | window shade |
| mcunet-person-det | Detector bbox | 1.24 s | PERSON conf=0.891 |
| mbv2-w0.35 (baseline) | ImageNet 1000 | 1.64 s | bannister |
| proxyless-w0.3 (baseline) | ImageNet 1000 | 2.51 s | obelisk |

Los modelos MCUNet corren correctamente en ESP32-S3 con resultados idénticos al host, pero son ~2× más lentos por pixel que MobileNetV2 en este runtime. La ventaja arquitectural de MCUNet depende de TinyEngine, que sólo soporta ARM Cortex-M. Sin él, MobileNetV2 es la opción práctica en ESP32-S3.

## Hardware

XIAO ESP32-S3 Sense (Xtensa LX7 dual @ 240 MHz, 8 MB PSRAM octal, 8 MB flash, OV2640).

## Estructura

```
host/                 Scripts PC (conda env "mcunet", Python 3.10)
├── env.yml
├── download_models.py        descarga .tflite a models/
├── eval_tflite.py            valida modelo en host
├── monitor.py                serial monitor no-interactivo
├── image_to_c.py             JPEG → C array int8
├── image_to_c_float.py       JPEG → C array float32
├── convert_to_c_array.sh     .tflite → C array
├── reproduce_host_inference.py
├── decode_camera_frame.py    recibe frame base64 del serial → PNG
├── generate_plots.py
├── visualize_inference.py
├── visualize_comparison.py
├── capture_and_regenerate.sh
└── sample_images/person_det.jpg

models/               .tflite preentrenados (no en git)
├── mcunet-vww2.tflite        941 KB
├── mcunet-in0.tflite         999 KB  (48×48 — descartado por tamaño)
├── mcunet-in2.tflite        1010 KB
├── mcunet-person-det.tflite  296 KB  (float32 input, 3 heads)
├── mbv2-w0.35.tflite         990 KB  (baseline)
└── proxyless-w0.3.tflite     992 KB  (baseline)

firmware/             6 proyectos ESP-IDF v5.2
├── vww_demo/                 HITO 1
├── imagenet_demo/            HITO 2 (mcunet-in2)
├── person_detect_demo/       HITO 3
├── benchmark_mbv2/           HITO 4 — MobileNetV2 baseline
├── benchmark_proxyless/      HITO 4 — ProxylessNAS baseline
├── camera_test/              diagnóstico cámara
└── i2c_scan/                 sweep de bus I²C + chip ID

docs/
├── REPORT.md
├── findings.md
├── CAMERA_FIX.md
├── camera_pinout_xiao.md
├── logs/                     capturas seriales
└── plots/                    9 plots PNG + frame capturado
```

## Setup desde cero

```bash
# Conda env para host (Python 3.10 + TF + torch)
conda env create -f host/env.yml
conda activate mcunet

# Bajar 6 modelos .tflite (~6 MB)
python host/download_models.py

# Instalar ESP-IDF v5.2 (~5 GB)
mkdir -p ~/esp && cd ~/esp
git clone -b v5.2 --recursive --depth 1 --shallow-submodules \
    https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh esp32s3
```

## Reproducir un demo

```bash
source ~/esp/esp-idf/export.sh
cd firmware/vww_demo
../../host/convert_to_c_array.sh ../../models/mcunet-vww2.tflite main/model_data.cc
python ../../host/image_to_c.py ../../host/sample_images/person_det.jpg 144x144 \
    main/test_image.cc g_test_image_vww
idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
python ../../host/monitor.py /dev/ttyACM0 30
```

## Bugs no obvios resueltos

1. `const` en C++ tiene internal linkage por defecto. Los arrays generados con `xxd -i` requieren `extern __attribute__((used)) const ...` dentro de `extern "C" { }`. Sin `extern`, el símbolo queda mangleado como `_ZL...` y el linker no lo encuentra.
2. `EXT_RAM_BSS_ATTR` requiere `CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY=y`. Si falta, el tensor arena se ubica en DRAM y desborda.
3. El stack default del `main_task` (3584 bytes) es insuficiente para `Invoke` de TFLite Micro: el chip se resetea silenciosamente después de la primera inferencia. Subir a 8 KB con `CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192`.
4. `idf.py monitor` pulsa DTR/RTS de una forma que el USB nativo del ESP32-S3 no maneja bien; `host/monitor.py` evita el problema.
5. Los inputs de los modelos MCUNet no son los esperables: `vww2` es 144×144 RGB (no 96×96 grayscale); `in0` es 48×48. Conviene confirmar con `host/eval_tflite.py` antes de flashear.
6. `MicroMutableOpResolver` requiere cada op del modelo registrada explícitamente. `Op for builtin opcode PAD not registered` es el error típico al olvidar uno.
7. Con el `test_image` embebido el binario excede `single_app_large` (1.5 MB). Custom `partitions.csv` con `factory 3 MB`.
8. `esp32-camera` v2.0.13 por default usa I²C port 1 para SCCB. En la combinación XIAO + Sense Expansion Board hay que forzar port 0 con `CONFIG_SCCB_HARDWARE_I2C_PORT0=y`.
9. El periférico HW i2c_master de IDF 5.2 hace busy-wait infinito en `s_i2c_send_commands` si la dirección no contesta ACK. Workaround: power-cycle físico del chip si quedó atascado tras flashes acumulados.

## Próximos pasos

- Re-correr TinyNAS con esp-nn como proxy de latencia para generar un modelo MCUNet adaptado al runtime real del ESP32-S3.
- Portar el experimento a una Grove Vision AI V2 (Cortex-M55 + NPU Ethos-U55), que sí es target oficial de TinyEngine.
- Fine-tunear los modelos sobre frames reales de la cámara para reducir la brecha OOD documentada arriba.

## Fuentes

- [mit-han-lab/mcunet](https://github.com/mit-han-lab/mcunet) (NeurIPS 2020)
- [mit-han-lab/tinyengine](https://github.com/mit-han-lab/tinyengine) (motor C/C++, sólo ARM)
- [espressif/esp-tflite-micro](https://components.espressif.com/components/espressif/esp-tflite-micro)
- [espressif/esp-nn](https://github.com/espressif/esp-nn)
- [Seeed XIAO ESP32-S3 Sense wiki](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/)

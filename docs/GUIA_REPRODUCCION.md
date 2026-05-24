# Guía de reproducción

Esta guía lleva a cualquier persona desde un repo recién clonado hasta tener los
modelos MCUNet corriendo en una XIAO ESP32-S3 Sense, viendo la salida por el
monitor serial. No asume conocimiento previo del proyecto. Sigue los pasos en
orden; cada bloque de comandos está pensado para copiar y pegar.

Tiempo aproximado la primera vez: 1–2 h (la mayor parte es bajar ESP-IDF, ~5 GB).

---

## 1. Qué vas a lograr

Al terminar tendrás:

- Un entorno host (PC Linux) capaz de validar los modelos `.tflite` y generar las
  imágenes/plots.
- ESP-IDF v5.2 instalado con el toolchain Xtensa para ESP32-S3.
- Cualquiera de los 6 firmwares compilado y flasheado en la placa, imprimiendo
  predicciones por el puerto serie.
- Opcionalmente, frames reales capturados por la cámara OV2640 y el GIF de demo.

---

## 2. Hardware necesario

| Componente | Detalle |
|---|---|
| XIAO ESP32-S3 **Sense** | La variante "Sense" trae la cámara OV2640 y PSRAM octal. La XIAO ESP32-S3 normal (sin cámara) sirve para los demos con imagen estática, pero no para los demos en vivo. |
| Cable USB-C de datos | No sirve uno solo de carga. La placa se expone como `/dev/ttyACM0`. |
| (Opcional) Sense Expansion Board | Útil pero no obligatorio. |

> Si tienes la antena Wi-Fi externa conectada y la cámara da problemas, prueba
> desconectándola: en algunas placas comparte líneas con el bus de la cámara.

---

## 3. Software necesario (host)

- Linux x86-64 (probado en Ubuntu 24.04 / kernel 6.8).
- `conda` o `miniconda` (para el entorno Python 3.10).
- `git`, `xxd` (viene en `vim-common`), `wget`.
- ~10 GB libres de disco (ESP-IDF + toolchain + entorno conda).

```bash
sudo apt update
sudo apt install -y git wget vim-common python3-venv
```

---

## 4. Clonar el repositorio

```bash
git clone https://github.com/JLDominguezM/MCUNetESP.git
cd MCUNetESP
```

Todos los comandos del resto de la guía asumen que estás dentro de `MCUNetESP/`.

---

## 5. Entorno host (Python + modelos)

Los modelos `.tflite` **no** están versionados en git (pesan ~6 MB en total y se
bajan del servidor del MIT). Se reconstruyen con un script.

```bash
# Crea el entorno conda "mcunet" (Python 3.10 + TF 2.15 + torch)
conda env create -f host/env.yml
conda activate mcunet

# Descarga los 6 modelos .tflite a models/
python host/download_models.py
```

Verifica que quedaron los 6 archivos:

```bash
ls -lh models/
# mbv2-w0.35.tflite  mcunet-in0.tflite  mcunet-in2.tflite
# mcunet-person-det.tflite  mcunet-vww2.tflite  proxyless-w0.3.tflite
```

Validación opcional en host (confirma que el modelo carga y predice antes de
gastar tiempo flasheando):

```bash
python host/eval_tflite.py models/mcunet-vww2.tflite host/sample_images/person_det.jpg
```

---

## 6. Instalar ESP-IDF v5.2

Sólo se hace una vez. Si ya tienes ESP-IDF v5.2 en otra ruta, salta este paso y
ajusta la ruta del `export.sh` en los siguientes.

```bash
mkdir -p ~/esp && cd ~/esp
git clone -b v5.2 --recursive --depth 1 --shallow-submodules \
    https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh esp32s3
cd ~/MCUNetESP    # vuelve al repo
```

A partir de aquí, **en cada terminal nueva** donde vayas a compilar firmware,
carga el entorno de ESP-IDF:

```bash
source ~/esp/esp-idf/export.sh
```

(Esto define `idf.py` y el toolchain. La activación del conda `mcunet` y el
`export.sh` de ESP-IDF pueden convivir en la misma terminal.)

---

## 7. Conectar la placa y dar permisos

Conecta la XIAO por USB-C. Comprueba que aparece:

```bash
ls /dev/ttyACM*       # debe listar /dev/ttyACM0
```

Si el flasheo falla con "Permission denied", agrega tu usuario al grupo
`dialout` (luego cierra sesión y vuelve a entrar, o reinicia):

```bash
sudo usermod -aG dialout "$USER"
```

> Si la placa no aparece como `/dev/ttyACM0`, revisa que el cable sea de datos y
> que no haya otro programa (otro `idf.py monitor`) ocupando el puerto.

---

## 8. Diagnóstico rápido de la placa (recomendado antes de los demos)

Antes de flashear un modelo, verifica que la placa y la cámara responden.

`i2c_scan` barre el bus I²C e imprime el chip ID del sensor de cámara
(`PID=0x26 VER=0x42` = OV2640):

```bash
source ~/esp/esp-idf/export.sh
cd firmware/i2c_scan
idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
python ../../host/monitor.py /dev/ttyACM0 15
cd ../..
```

Si ves el chip ID, la cámara está viva. Si no responde nada coherente, ve a la
sección **Solución de problemas → cámara**.

---

## 9. Reproducir un demo (patrón general)

Cada firmware sigue el mismo patrón de 5 pasos:

1. Cargar el entorno ESP-IDF.
2. Convertir el `.tflite` a un C array (`model_data.cc`).
3. Generar la imagen de prueba embebida (`test_image.cc`).
4. Compilar y flashear.
5. Leer el serial con `host/monitor.py`.

Los pasos 2 y 3 sólo hace falta repetirlos si cambias de modelo o de imagen.

### Tabla de comandos por demo

Cada celda "modelo" y "imagen" reemplaza las líneas 2 y 3 del patrón. Todos los
comandos se ejecutan **desde el directorio del firmware** (`cd firmware/<nombre>`).

| Demo | Modelo (`convert_to_c_array.sh`) | Imagen (`image_to_c*.py`) | Input | Símbolo imagen |
|---|---|---|---|---|
| `vww_demo` | `mcunet-vww2.tflite` | `image_to_c.py` | `144x144` | `g_test_image_vww` |
| `imagenet_demo` | `mcunet-in2.tflite` | `image_to_c.py` | `160x160` | `g_test_image_in` |
| `benchmark_mbv2` | `mbv2-w0.35.tflite` | `image_to_c.py` | `144x144` | `g_test_image_in` |
| `benchmark_proxyless` | `proxyless-w0.3.tflite` | `image_to_c.py` | `176x176` | `g_test_image_in` |
| `person_detect_demo` | `mcunet-person-det.tflite` | `image_to_c_float.py` | `160x128` | `g_test_image_pd` |

### Ejemplo completo — vww_demo (HITO 1)

```bash
source ~/esp/esp-idf/export.sh
cd firmware/vww_demo

../../host/convert_to_c_array.sh ../../models/mcunet-vww2.tflite main/model_data.cc
python ../../host/image_to_c.py ../../host/sample_images/person_det.jpg 144x144 \
    main/test_image.cc g_test_image_vww

idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
python ../../host/monitor.py /dev/ttyACM0 30

cd ../..
```

Salida esperada (clasificador persona / no-persona):

```
I vww: model OK: in=[1,144,144,3]  arena=473644 / 614400 bytes
I vww: PERSON   scores=[ -51,  51]
```

### Ejemplo completo — imagenet_demo (HITO 2)

```bash
source ~/esp/esp-idf/export.sh
cd firmware/imagenet_demo

../../host/convert_to_c_array.sh ../../models/mcunet-in2.tflite main/model_data.cc
python ../../host/image_to_c.py ../../host/sample_images/person_det.jpg 160x160 \
    main/test_image.cc g_test_image_in

idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
python ../../host/monitor.py /dev/ttyACM0 30

cd ../..
```

Imprime los índices de las 5 mejores clases. Para mapear índices a nombres:

```bash
wget https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
awk 'NR==907' ImageNetLabels.txt   # índice 906 -> "window shade"
```

> Para `benchmark_mbv2`, `benchmark_proxyless` y `person_detect_demo` usa la
> misma estructura sustituyendo modelo/imagen/símbolo según la tabla de arriba.
> `person_detect_demo` usa `image_to_c_float.py` (input float32) y corre sólo
> sobre la imagen estática, no en vivo.

### Comandos extra útiles

```bash
idf.py build                 # sólo compila, sin flashear
idf.py -p /dev/ttyACM0 monitor   # monitor interactivo de ESP-IDF (Ctrl+] sale)
idf.py fullclean             # borra build/ si algo quedó en estado raro
```

`host/monitor.py` es un monitor no-interactivo que captura N segundos y termina
solo — es lo que conviene para guardar logs. El `idf.py monitor` interactivo a
veces pulsa DTR/RTS de forma que el USB nativo del S3 no maneja bien.

---

## 10. Demo en vivo con la cámara

Cuatro firmwares (`vww_demo`, `imagenet_demo`, `benchmark_mbv2`,
`benchmark_proxyless`) detectan la cámara al arrancar: si inicializa, clasifican
frames en tiempo real; si falla, caen a la imagen estática embebida.

Flashea como en el paso 9 y deja el monitor corriendo más tiempo (p. ej. 60 s)
mientras apuntas la cámara a distintas escenas:

```bash
python host/monitor.py /dev/ttyACM0 60
```

---

## 11. Capturar frames reales y regenerar el GIF de demo

El firmware `camera_test` captura 5 frames consecutivos en RGB565 y los envía por
serial en base64. El script host los recibe, corre `mcunet-vww2` sobre cada uno y
compone `docs/plots/demo_visual.gif`.

```bash
source ~/esp/esp-idf/export.sh
cd firmware/camera_test
idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
cd ../..

# En el entorno conda (necesita TF + pillow + pyserial):
conda activate mcunet
python host/decode_camera_frames_multi.py /dev/ttyACM0
```

El firmware espera 1.5 s entre frames para que el sujeto se reposicione. El GIF y
los PNG individuales quedan en `docs/plots/demo_visual*.{gif,png}`.

> Si quieres una sola captura estática en lugar de la secuencia, usa
> `host/decode_camera_frame.py`.

---

## 12. Validar que el ESP32 coincide con el host

Para confirmar que la inferencia del chip es bit-idéntica a TFLite en PC:

```bash
conda activate mcunet
python host/reproduce_host_inference.py
```

Y para regenerar los 9 plots del reporte:

```bash
python host/generate_plots.py
```

---

## 13. Solución de problemas

### El linker no encuentra `g_mcunet_...` (`_ZL...` undefined)

En C++ `const` tiene *internal linkage* por defecto. Los arrays generados deben
declararse `extern __attribute__((used)) const ...` dentro de `extern "C" {}`.
`convert_to_c_array.sh` ya lo hace; si editaste un `.cc` a mano, revísalo.

### El chip se resetea después de la primera inferencia

Stack del `main_task` insuficiente. Ya está corregido con
`CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192` en `sdkconfig.defaults`. Si copiaste un
proyecto sin ese flag, agrégalo.

### `Op for builtin opcode PAD not registered` (u otro op)

Falta registrar un operador en el `MicroMutableOpResolver`. Cada op que use el
modelo debe añadirse explícitamente en el `main.cc` del firmware.

### El binario no cabe en la partición

Con la imagen de prueba embebida el binario supera `single_app_large`. Los
proyectos ya traen un `partitions.csv` con una partición `factory` de 3 MB; no lo
borres.

### Cámara: "camera not supported" / no inicializa

Causas conocidas, en orden de probabilidad:

1. **Puerto SCCB equivocado.** `esp32-camera` v2.0.13 usa I²C port 1 por
   defecto; en esta placa hay que forzar port 0. Ya está en `sdkconfig.defaults`
   como `CONFIG_SCCB_HARDWARE_I2C_PORT0=y`. Confírmalo si copiaste un proyecto.
2. **Chip atascado tras varios flashes.** El periférico i2c_master de IDF 5.2
   puede quedar en busy-wait infinito si una dirección no contesta ACK.
   Solución: **power-cycle físico** — desconecta y reconecta el USB, luego
   reflashea. A veces hace falta hacerlo un par de veces.
3. **Antena Wi-Fi conectada.** Desconéctala y deja sólo la cámara.

Para confirmar que la cámara está físicamente viva, corre `i2c_scan` (paso 8) y
busca `PID=0x26 VER=0x42`.

### `Permission denied` al flashear

Falta el grupo `dialout` (paso 7) o hay otro proceso usando `/dev/ttyACM0`.

---

## 14. Referencia rápida de directorios

```
host/        Scripts de PC (entorno conda "mcunet")
models/      Modelos .tflite (se bajan con download_models.py; no en git)
firmware/    6 proyectos ESP-IDF v5.2
docs/        Reporte, findings, logs y plots
```

Más detalle técnico del experimento en [`REPORT.md`](REPORT.md) y
[`findings.md`](findings.md). Diagnóstico de cámara en
[`CAMERA_FIX.md`](CAMERA_FIX.md) y pinout en
[`camera_pinout_xiao.md`](camera_pinout_xiao.md).

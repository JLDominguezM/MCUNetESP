# Procedimiento para destrabar la cámara OV2640 del XIAO ESP32-S3 Sense

## Síntoma

```
E camera: Detected camera not supported.
E camera: Camera probe failed with error 0x106 (ESP_ERR_NOT_SUPPORTED)
```

El driver `esp32-camera` lee el chip ID del sensor por SCCB (I²C) y la respuesta no concuerda con ningún sensor conocido. Las dos causas habituales son:

1. El bus SCCB no está enrutado al port I²C correcto.
2. El periférico HW i2c_master de IDF 5.2 está atascado de operaciones previas.

Sin necesidad de tocar hardware, la mayoría de los casos se resuelven con los dos puntos abajo.

## Fix de configuración (la causa #1)

Añadir en `sdkconfig.defaults`:

```
CONFIG_SCCB_HARDWARE_I2C_PORT0=y
# CONFIG_SCCB_HARDWARE_I2C_PORT1 is not set
```

El default del componente `esp32-camera` 2.0.13 es I²C port 1, que en la combinación XIAO ESP32-S3 + Sense Expansion Board no funciona. Forzar port 0 destraba el SCCB para esta combinación.

`firmware/camera_test/sdkconfig.defaults` ya tiene este setting.

## Reset del periférico I²C (la causa #2)

El periférico HW i2c_master de IDF 5.2 tiene un busy-wait infinito en `s_i2c_send_commands` (line 424 de `driver/i2c/i2c_master.c`) cuando una operación previa quedó incompleta. El reset por DTR/RTS de `idf.py flash` no recupera el periférico — sólo desconectar VCC lo limpia.

Procedimiento:

1. Desconectar el USB-C del XIAO. Esperar 5 segundos para que se descarguen los capacitores.
2. Reconectar el USB-C.
3. Re-correr el firmware (`idf.py monitor` o `host/decode_camera_frame.py`).

Si después del power-cycle el log dice `Detected OV2640 camera`, la cámara está funcionando.

## Verificación con `firmware/i2c_scan`

Si el fix anterior no destraba el bus, hay un firmware de diagnóstico estandalone que no depende del componente `esp32-camera`:

```bash
source ~/esp/esp-idf/export.sh
cd firmware/i2c_scan
idf.py -p /dev/ttyACM0 flash
cd ../..
python host/monitor.py /dev/ttyACM0 10
```

Genera XCLK en GPIO10 con LEDC y hace un sweep del bus I²C (`i2c_master_probe` para cada dirección 0x01–0x7F). Si la cinta FPC está bien conectada, se ve la respuesta en `0x30`:

```
30: 30 -- -- -- -- -- -- -- ...
```

Y la lectura del chip ID (registros 0x0A y 0x0B del bank=1 del OV2640):

```
I chipid: OV2640 protocol  → write 0xFF=0x01: OK
I chipid:                  → read  0x0A (PID): OK value=0x26
I chipid:                  → read  0x0B (VER): OK value=0x42
I chipid:   Combined ID = 0x2642
I chipid:   >>> Es un OV2640 (variante 0x2642) <<<
```

Si el sweep está totalmente vacío incluso con XCLK, hay un problema físico (cinta FPC, conector ZIF o sensor) que sí requiere intervención de hardware.

## Capturar un frame de prueba

Una vez la cámara responde, `firmware/camera_test` prueba varias configs y captura un frame:

```bash
cd firmware/camera_test
source ~/esp/esp-idf/export.sh
idf.py -p /dev/ttyACM0 flash
cd ../..
python host/decode_camera_frame.py /dev/ttyACM0 30
```

Salida esperada:

```
I camera: Detected OV2640 camera
I cam_test: >>> SUCCESS with 20MHz JPEG QVGA
I cam_test: captured frame: 320x240  fmt=4  len=12345 bytes
saved docs/plots/sample_camera_frame.jpg
converted to docs/plots/sample_camera_frame.png
```

## Regenerar la documentación con la imagen capturada

```bash
./host/capture_and_regenerate.sh
```

El script:

1. Re-cuantiza la imagen al input de cada modelo (144×144, 160×160, 176×176, 128×160 fp32).
2. Re-bakea `test_image.cc` en los cinco firmware (vww, imagenet, person_detect, mbv2, proxyless).
3. Re-flashea y captura logs.
4. Re-genera los plots con la imagen capturada.

`visualize_inference.py` detecta automáticamente la presencia de `docs/plots/sample_camera_frame.png` y la usa en lugar de la imagen del repo MCUNet.

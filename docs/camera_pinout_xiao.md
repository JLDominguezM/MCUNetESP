# XIAO ESP32-S3 Sense — Pinout de cámara OV2640

Referencia: [Seeed wiki — XIAO ESP32-S3 Sense](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/).

La cámara OV2640 va conectada al expansion board interno por el conector FPC.
Pines mapeados al driver `esp32-camera` (Espressif):

```c
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40   // SDA cámara (no usar para I2C user)
#define SIOC_GPIO_NUM     39   // SCL cámara

#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13
```

## Notas

- Necesitas habilitar PSRAM (octal 80 MHz) o el driver no puede asignar framebuffer >96×96.
- Para VWW el modelo espera grayscale 96×96 → setea `pixel_format = PIXFORMAT_GRAYSCALE` y `frame_size = FRAMESIZE_96X96`.
- Para mcunet-in0 (160×160 RGB) → `pixel_format = PIXFORMAT_RGB565` y luego conviertes a uint8 RGB888 antes de cuantizar.
- USB-C nativo del XIAO se enumera como `/dev/ttyACM0` en Linux (no `ttyUSB0`).

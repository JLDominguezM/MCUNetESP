# person_detect_demo — MCUNet person detection (128×160 float32)

Despliega `mcunet-person-det.tflite` (descargado por `host/download_models.py`) en el XIAO ESP32-S3 Sense.

El modelo es un detector tipo SSD/YOLO con tres cabezas multi-escala:

- `output[0]`: (1, 4, 5, 18) — stride 32
- `output[1]`: (1, 8, 10, 18) — stride 16
- `output[2]`: (1, 16, 20, 18) — stride 8

18 valores por celda = 3 anchors × (x, y, w, h, obj, person_logit).

El demo aplica `sigmoid(obj) × sigmoid(person_logit)` por celda y reporta el arg-max de las tres cabezas. Sin NMS y sin bbox final dibujado — el objetivo es validar inferencia, no producción.

Sobre la imagen de prueba del repo MCUNet (`host/sample_images/person_det.jpg`), la cabeza media (stride 16, 8×10 grid) detecta persona con confianza 0.891. Latencia: 1.24 s ± 4 µs.

## Build y flash

```bash
source ~/esp/esp-idf/export.sh
../../host/convert_to_c_array.sh ../../models/mcunet-person-det.tflite main/model_data.cc
python ../../host/image_to_c_float.py ../../host/sample_images/person_det.jpg 160x128 \
    main/test_image.cc g_test_image_pd
idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
python ../../host/monitor.py /dev/ttyACM0 30
```

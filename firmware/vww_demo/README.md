# vww_demo — mcunet-vww2 (Visual Wake Words)

Despliega `mcunet-vww2.tflite` (clasificador binario persona / no-persona) en el XIAO ESP32-S3 Sense.

| | valor |
|---|---|
| Input | 144×144×3 int8 |
| Output | 1×2 logits int8 ([no-pers, person]) |
| Tensor arena | 600 KB en PSRAM |
| Bin size | ~1.42 MB |
| Latencia medida | 3.39 s ± 1 ms |
| Predicción sobre `person_det.jpg` | PERSON, scores [-51, +51] |

Coincide bit-a-bit con la inferencia del host TFLite Interpreter.

Sobre la imagen capturada por la propia cámara del Sense (`docs/plots/sample_camera_frame.png`) el modelo predice `no-pers` con scores [+22, -22]. La diferencia entre las dos predicciones cuantifica una brecha de generalización ante imágenes con orientación, iluminación y composición distintas al training set.

## Build, flash y monitor

```bash
source ~/esp/esp-idf/export.sh
../../host/convert_to_c_array.sh ../../models/mcunet-vww2.tflite main/model_data.cc
python ../../host/image_to_c.py ../../host/sample_images/person_det.jpg 144x144 \
    main/test_image.cc g_test_image_vww
idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
python ../../host/monitor.py /dev/ttyACM0 30
```

El firmware tiene fallback automático: si la cámara OV2640 inicializa correctamente captura frames en vivo; si falla, usa la imagen estática embebida en `test_image.cc`.

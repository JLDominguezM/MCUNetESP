# benchmark_mbv2 — MobileNetV2-w0.35 baseline

Despliega `mbv2-w0.35.tflite` (ImageNet 1000 clases, baseline para comparar contra `mcunet-in2`) en el XIAO ESP32-S3 Sense.

| | valor |
|---|---|
| Input | 144×144×3 int8 |
| Output | 1×1000 logits int8 |
| Tensor arena | 700 KB en PSRAM |
| Bin size | ~1.43 MB |
| Latencia medida | 1.64 s ± 0.01 ms |
| Top-1 sobre `person_det.jpg` | clase 422 ("bannister") |

A igual número de píxeles de entrada que `mcunet-vww2` (144×144), corre ~2× más rápido. El demo está instrumentado con `MicroProfiler` para el desglose por op.

## Build, flash y monitor

```bash
source ~/esp/esp-idf/export.sh
../../host/convert_to_c_array.sh ../../models/mbv2-w0.35.tflite main/model_data.cc
python ../../host/image_to_c.py ../../host/sample_images/person_det.jpg 144x144 \
    main/test_image.cc g_test_image_in
idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
python ../../host/monitor.py /dev/ttyACM0 30
```

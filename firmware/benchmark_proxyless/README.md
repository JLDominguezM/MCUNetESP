# benchmark_proxyless — ProxylessNAS-w0.3 baseline

Despliega `proxyless-w0.3.tflite` (ImageNet 1000 clases, segundo baseline NAS para comparar contra `mcunet-in2`) en el XIAO ESP32-S3 Sense.

| | valor |
|---|---|
| Input | 176×176×3 int8 |
| Output | 1×1000 logits int8 |
| Tensor arena | 700 KB en PSRAM |
| Bin size | ~1.45 MB |
| Latencia medida | 2.51 s ± 0.01 ms |
| Top-1 sobre `person_det.jpg` | clase 683 ("obelisk") |

A pesar de que ProxylessNAS-w0.3 también proviene de un NAS, su latencia por píxel (~0.08 ms/px) es similar a MobileNetV2 — y MCUNet (~0.15 ms/px) es 2× más lento. El experimento del HITO 4 lo cuantifica.

## Build, flash y monitor

```bash
source ~/esp/esp-idf/export.sh
../../host/convert_to_c_array.sh ../../models/proxyless-w0.3.tflite main/model_data.cc
python ../../host/image_to_c.py ../../host/sample_images/person_det.jpg 176x176 \
    main/test_image.cc g_test_image_in
idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
python ../../host/monitor.py /dev/ttyACM0 30
```

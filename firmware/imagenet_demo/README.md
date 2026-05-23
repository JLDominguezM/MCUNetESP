# imagenet_demo — mcunet-in2 sobre XIAO ESP32-S3 Sense

Despliega `mcunet-in2.tflite` (ImageNet 1000 clases) en el XIAO ESP32-S3 Sense usando `esp-tflite-micro` + `esp-nn`.

| | valor |
|---|---|
| Input | 160×160×3 int8 |
| Output | 1×1000 logits int8 |
| Tensor arena | 700 KB en PSRAM |
| Bin size | ~1.52 MB |
| Latencia medida | 3.93 s ± 0.01 ms |
| Top-1 sobre `person_det.jpg` | clase 906 ("window shade") |

Coincide bit-a-bit con la inferencia del host TFLite Interpreter. El demo está instrumentado con `tflite::MicroProfiler::LogTicksPerTagCsv()` y reporta el desglose de ops en la primera inferencia.

## Build, flash y monitor

```bash
source ~/esp/esp-idf/export.sh
../../host/convert_to_c_array.sh ../../models/mcunet-in2.tflite main/model_data.cc
python ../../host/image_to_c.py ../../host/sample_images/person_det.jpg 160x160 \
    main/test_image.cc g_test_image_in
idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 flash
python ../../host/monitor.py /dev/ttyACM0 30
```

El demo imprime los índices de las 5 mejores clases en cada inferencia. Para mapear los índices a nombres usar el archivo de labels de ImageNet:

```
wget https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
awk 'NR==907' ImageNetLabels.txt   # → window shade
```

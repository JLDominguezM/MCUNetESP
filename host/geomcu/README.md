# geomcu host pipeline

Conversión y evaluación del modelo `microwd_paper` (MiCrowdNet de
`CentroFuturoCiudades/geomcu-counting`) para deploy en ESP32-S3. El
modelo se entrenó en ShanghaiTech Part B sobre el server. Aquí está
todo lo que corre en la PC: reproducir la arquitectura, validar contra
el checkpoint original, cuantizar a int8 y medir cuánto se pierde.

El run de referencia es `microwd_paper_original_adaptive_mse_scaled_count001_softplus_seed7`
(MAE 16.83 sobre el test set Part B, 316 imágenes).

## Workspace local

El checkpoint y el dataset pesan ~200 MB y son privados, no van en git.
Viven en `~/work/geomcu-deploy/`:

```
~/work/geomcu-deploy/
├── run/
│   ├── best_mae.pt
│   ├── eval_best_mae.json
│   └── detailed_eval/per_image_predictions.csv
├── dataset/part_B/
│   ├── test_data/{images, annotations, ground-truth}/   # 316 imgs
│   └── train_data/{images, annotations, ground-truth}/  # 400 imgs
└── tflite/
```

Para re-sincronizar desde el server `sophie`:

```bash
rsync -ah sophie:Sophie/geomcu-counting/student/runs_shanghai_b/microwd_paper_original_adaptive_mse_scaled_count001_softplus_seed7/ \
    ~/work/geomcu-deploy/run/
rsync -ah sophie:Sophie/geomcu-counting/datasets/ShanghaiTech/part_B/ \
    ~/work/geomcu-deploy/dataset/part_B/
```

## Archivos

| Archivo | Qué hace |
|---|---|
| `model_microwd.py` | Reimplementación PyTorch standalone de `MiCrowdNetPaperFullFrame`. Mismos nombres de submódulos para que `load_state_dict(strict=True)` funcione sin remapeo. |
| `model_microwd_keras.py` | Port Keras 1:1. Permite cuantizar con `TFLiteConverter.from_keras_model()` sin pasar por ONNX (que dio bugs con kernels pares 16×16). |
| `port_weights.py` | Copia pesos del .pt al modelo Keras. Verifica parity con un forward sobre IMG_1 (PyTorch ≡ Keras a 4e-8). |
| `forward_check.py` | Sanity check del float32 contra `per_image_predictions.csv`. Espera offset sistemático ~0.1% por GPU/TF32 vs CPU/FP32. |
| `convert_tflite.py` | Genera 5 variants TFLite: fp32, int8 (I/O int8), int8 + fp32 output, dynamic range, y logits_int8 (sin softplus). Soporta `--input-shape` para deploy patch-based. |
| `eval_tflite.py` | Evalúa los 5 variants sobre N imgs del test set. Maneja softplus post-process para el variant logits_int8. |
| `list_ops.py` | Lista ops únicos por TFLite. Útil para dimensionar el `MicroMutableOpResolver` del firmware. |
| `patch_proto.py` | Prototipo de inferencia patch-based (MCUNetV2 style). Tile la imagen en N×M patches con halo, corre el TFLite resized por patch, stitchea. Mide la pérdida de hacer patch-based vs full forward. |
| `export_onnx.py` | Export ONNX (sólo para debug, no se usa en el pipeline TFLite final). |

## Resultados de cuantización

Evaluación sobre 50 imágenes del test set, comparando contra el CSV del
repo (corrió en GPU fp32, MAE 15.27 esperada).

```
Model                                MAE     RMSE   mean_pred   mean_true
-------------------------------------------------------------------------
PT (CSV ref)                      15.267   22.643     120.218     126.640
TFLite fp32                       15.311   22.718     120.092     126.640
TFLite int8 (I/O)                 53.422   59.855      73.218     126.640
TFLite int8 + fp32 out            53.422   59.855      73.218     126.640
TFLite dyn-range                  15.270   22.501     120.318     126.640
TFLite logits_int8 + fp32 softplus 14.555  20.856     123.604     126.640
```

Lecturas:

1. **fp32 TFLite** coincide con PyTorch a 4e-8. La conversión
   Keras→TFLite no introduce error.
2. **int8 puro** colapsa: MAE 53 (3.5× peor). El TFLite Converter
   implementa `softplus` como `EXP` + `LOG`. EXP produce valores en
   [4.5e-5, 148] que no caben en int8 con resolución útil. Casi todas
   las densidades quedan redondeadas a 0. La hipótesis "el problema es
   el output quantization scale" fue descartada porque el variant
   **int8 + fp32 out** da exactamente lo mismo (53.42 MAE): el daño
   está adentro, no en la salida.
3. **dynamic range** (pesos int8, activaciones fp32) mantiene la
   accuracy pero no aprovecha esp-nn en el chip. Sirve si la latencia
   no importa.
4. **logits_int8 + fp32 softplus** es el ganador. Saca el `softplus`
   del grafo (entrenando el modelo Keras con
   `drop_final_activation=True`), cuantiza el resto, aplica `softplus`
   en fp32 después del dequant. Los logits crudos están en rango
   [-11, +5], muy amigable para int8. Resultado: MAE 14.55, ligeramente
   mejor que el fp32 ref (el int8 actúa como regularizador suave).

El modelo `logits_int8` tampoco requiere `EXP`/`LOG`/`SOFTPLUS` en el
firmware, así que el `MicroMutableOpResolver` es `<5>`: `CONV_2D`,
`DEPTHWISE_CONV_2D`, `MAX_POOL_2D`, `ADD`, `CONCATENATION`.

## Patch-based en host

`patch_proto.py` valida el harness antes de portarlo a C++ en el chip.
Tile la imagen en N×M patches con halo, resizea el input del
Interpreter y corre uno por uno. Resultado típico con 2×2 + halo 96:

```
=== IMG_1 ===
  count_full = 25.4181  count_patched = 22.5079  delta -2.91 (-11.45%)
=== IMG_10 ===
  count_full = 191.5640 count_patched = 179.7369 delta -11.83 (-6.17%)
=== IMG_100 ===
  count_full = 126.5291 count_patched = 124.3151 delta -2.21 (-1.75%)
=== IMG_101 ===
  count_full = 41.2447  count_patched = 35.5585  delta -5.69 (-13.79%)
=== IMG_102 ===
  count_full = 68.7206  count_patched = 63.5881  delta -5.13 (-7.47%)
```

Promedio: -8% per image. La pérdida es del patching (border effects),
no de la cuantización. El modelo no fue entrenado patch-aware como
recomienda MCUNetV2. La fracción de error depende fuertemente de
cuántas personas caen cerca del borde del patch.

Probé `mode='edge'` vs `mode='constant'` para el padding del halo.
Edge da -8% promedio, constant (cero) da -27%. El modelo se entrenó
con `ZeroPadding2D` interno, pero el padding EXTERNO que añadimos en
el harness debe replicar (no rellenar con cero), porque las
"extensiones" artificiales con cero confunden al modelo más que la
replicación natural de píxeles del borde.

## Cómo correrlo

```bash
conda activate mcunet     # torch 2.2.2 + tf 2.15 + numpy 1.26
python forward_check.py   # PT reimpl vs CSV ref a 0.5%
python convert_tflite.py  # genera los 5 variants
python eval_tflite.py     # tabla MAE
python list_ops.py        # ops por TFLite
python patch_proto.py     # patched vs full sobre fp32
```

La cuantización int8 calibra con 100 forwards full-res. En CPU local
toma unos 17 minutos; en el server `sophie` (64 cores) ~3 minutos al
usar crop más pequeño para el patch-deploy variant.

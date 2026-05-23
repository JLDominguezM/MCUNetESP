// VWW demo: mcunet-vww2 sobre XIAO ESP32-S3 Sense.
//
// Pipeline:
//   1. OV2640 -> 144x144 RGB int8 (vía esp32-camera)
//   2. TFLite Micro invoke (CONV / DWCONV optimizados por esp-nn)
//   3. log latencia + clase (0=no-person, 1=person)
//
// Si la cámara no responde (ESP_ERR_NOT_SUPPORTED), entra en MODO ESTÁTICO:
// usa g_test_image_vww (imagen 144x144 RGB embebida en flash) en bucle.
// Esto permite medir latencia/heap/accuracy aunque la cámara esté desconectada.
#include <cstdio>
#include <cstdint>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "image_provider.h"

extern "C" const unsigned char g_mcunet_vww2[];
extern "C" const unsigned int g_mcunet_vww2_len;

extern "C" const int8_t g_test_image_vww[];
extern "C" const int g_test_image_vww_w;
extern "C" const int g_test_image_vww_h;

namespace {
// mcunet-vww2 acepta 144x144x3 int8 (verificado con eval_tflite.py).
constexpr int kInputW = 144;
constexpr int kInputH = 144;
constexpr int kArenaBytes = 600 * 1024;

EXT_RAM_BSS_ATTR uint8_t tensor_arena[kArenaBytes];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

const char* TAG = "vww";
}  // namespace

static void SetupModel() {
    if (g_mcunet_vww2_len < 100) {
        ESP_LOGE(TAG, "model_data.cc has placeholder model — bake the real .tflite first");
        vTaskDelay(portMAX_DELAY);
    }
    model = tflite::GetModel(g_mcunet_vww2);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "schema mismatch: model=%lu lib=%d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        vTaskDelay(portMAX_DELAY);
    }

    // Ops verificados con eval_tflite.py: CONV, DWCONV, ADD, AVG_POOL, PAD, RESHAPE.
    // Añadimos también otros comunes (FC, SOFTMAX, MAX_POOL, MUL, MEAN, QUANT) por
    // robustez en caso de cambiar a otro modelo de la familia MCUNet.
    static tflite::MicroMutableOpResolver<14> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddMean();
    resolver.AddPad();
    resolver.AddConcatenation();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaBytes);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (arena too small?)");
        vTaskDelay(portMAX_DELAY);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    ESP_LOGI(TAG, "model OK: in=[%d,%d,%d,%d] type=%d  out=[%d,%d] type=%d",
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3], input->type,
             output->dims->data[0], output->dims->data[1], output->type);
    ESP_LOGI(TAG, "arena used: %d / %d bytes", interpreter->arena_used_bytes(), kArenaBytes);
}

extern "C" void app_main() {
    ESP_LOGI(TAG, "MCUNet-VWW demo on XIAO ESP32-S3 Sense");
    ESP_LOGI(TAG, "free heap internal: %d   PSRAM: %d",
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    const bool camera_ok = (ImageProviderInit() == ESP_OK);
    if (!camera_ok) {
        ESP_LOGW(TAG, "camera init failed — entering STATIC IMAGE mode");
        ESP_LOGW(TAG, "test image dims: %dx%d", g_test_image_vww_w, g_test_image_vww_h);
    }
    SetupModel();

    int64_t lat_sum_us = 0;
    int frames = 0;
    while (true) {
        if (camera_ok) {
            if (GetImage(input->data.int8, kInputW, kInputH) != ESP_OK) {
                vTaskDelay(pdMS_TO_TICKS(100));
                continue;
            }
        } else {
            // Static fallback: cargar imagen embebida directamente al tensor input.
            memcpy(input->data.int8, g_test_image_vww, kInputW * kInputH * 3);
        }
        {
        int64_t t0 = esp_timer_get_time();
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed");
            continue;
        }
        int64_t dt = esp_timer_get_time() - t0;
        lat_sum_us += dt;
        ++frames;

        // mcunet-vww output: 2 logits int8 (no-person, person).
        int8_t s0 = output->data.int8[0];
        int8_t s1 = output->data.int8[1];
        const char* label = (s1 > s0) ? "PERSON " : "no-pers";
        ESP_LOGI(TAG, "%s  scores=[%4d,%4d]  inv=%lld us  avg=%lld us",
                 label, s0, s1, dt, lat_sum_us / frames);

        if (frames % 100 == 0) {
            ESP_LOGI(TAG, ">>> after %d frames: avg %lld us, min_free_int=%d, min_free_psram=%d",
                     frames, lat_sum_us / frames,
                     heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
                     heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
        }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

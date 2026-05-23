// ImageNet demo: mcunet-in2 (160x160 RGB, 1000 clases) sobre XIAO ESP32-S3 Sense.
// Si la cámara no responde, usa imagen estática embebida (modo benchmark).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "image_provider.h"

extern "C" const unsigned char g_mbv2_w0_35[];
extern "C" const unsigned int g_mbv2_w0_35_len;
extern "C" const int8_t g_test_image_in[];
extern "C" const int g_test_image_in_w;
extern "C" const int g_test_image_in_h;

namespace {
constexpr int kInputW = 144;
constexpr int kInputH = 144;
constexpr int kArenaBytes = 700 * 1024;

EXT_RAM_BSS_ATTR uint8_t tensor_arena[kArenaBytes];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroProfiler* profiler = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
const char* TAG = "mbv2";
}  // namespace

static void SetupModel() {
    if (g_mbv2_w0_35_len < 100) {
        ESP_LOGE(TAG, "model_data.cc has placeholder — bake the real .tflite first");
        vTaskDelay(portMAX_DELAY);
    }
    model = tflite::GetModel(g_mbv2_w0_35);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "schema mismatch");
        vTaskDelay(portMAX_DELAY);
    }
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

    static tflite::MicroProfiler static_profiler;
    profiler = &static_profiler;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaBytes, nullptr, profiler);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (arena too small?)");
        vTaskDelay(portMAX_DELAY);
    }
    input = interpreter->input(0);
    output = interpreter->output(0);
    ESP_LOGI(TAG, "model OK: in=[%d,%d,%d,%d]  out=[%d,%d]  arena=%d/%d",
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3],
             output->dims->data[0], output->dims->data[1],
             interpreter->arena_used_bytes(), kArenaBytes);
}

extern "C" void app_main() {
    ESP_LOGI(TAG, "MobileNetV2-w0.35 ImageNet demo");
    ESP_LOGI(TAG, "free heap internal: %d   PSRAM: %d",
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    const bool camera_ok = (ImageProviderInit() == ESP_OK);
    if (!camera_ok) {
        ESP_LOGW(TAG, "camera init failed — STATIC mode (%dx%d)",
                 g_test_image_in_w, g_test_image_in_h);
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
            memcpy(input->data.int8, g_test_image_in, kInputW * kInputH * 3);
        }
        // Limpiamos eventos antes de cada invoke. Tag duraran sólo 1 frame.
        profiler->ClearEvents();
        int64_t t0 = esp_timer_get_time();
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed");
            continue;
        }
        int64_t dt = esp_timer_get_time() - t0;
        lat_sum_us += dt;
        ++frames;

        // Imprimir el desglose por op SOLO en el primer frame (los siguientes son idénticos).
        if (frames == 1) {
            ESP_LOGI(TAG, "===== PROFILER CSV BEGIN =====");
            profiler->LogTicksPerTagCsv();
            ESP_LOGI(TAG, "===== PROFILER CSV END =====");
        }

        // top-5 over 1000 int8 logits
        const int N = output->dims->data[1];
        int idx[5] = {0, 1, 2, 3, 4};
        for (int i = 5; i < N; ++i) {
            int min_pos = 0;
            for (int k = 1; k < 5; ++k) {
                if (output->data.int8[idx[k]] < output->data.int8[idx[min_pos]]) min_pos = k;
            }
            if (output->data.int8[i] > output->data.int8[idx[min_pos]]) idx[min_pos] = i;
        }
        std::sort(idx, idx + 5, [&](int a, int b) {
            return output->data.int8[a] > output->data.int8[b];
        });
        ESP_LOGI(TAG, "top5=[%d,%d,%d,%d,%d]  scores=[%d,%d,%d,%d,%d]  inv=%lld us  avg=%lld us",
                 idx[0], idx[1], idx[2], idx[3], idx[4],
                 output->data.int8[idx[0]], output->data.int8[idx[1]],
                 output->data.int8[idx[2]], output->data.int8[idx[3]], output->data.int8[idx[4]],
                 dt, lat_sum_us / frames);
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

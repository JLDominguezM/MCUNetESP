// Person detection demo: mcunet-person-det (128x160 RGB float32) en XIAO ESP32-S3 Sense.
//
// El modelo es un detector tipo YOLO/SSD con 3 cabezas multi-escala:
//   output[0]: (1, 4, 5, 18)   stride 32 — objetos grandes
//   output[1]: (1, 8, 10, 18)  stride 16 — objetos medianos
//   output[2]: (1, 16, 20, 18) stride 8  — objetos pequeños
// 18 = 3 anchors × (x, y, w, h, obj_logit, person_logit)
//
// Demo simple: para cada cabeza, encontrar la celda con mayor obj_logit y
// reportarla. Sin NMS — esto valida que el modelo corre, no es production-grade.
//
// Modo estático con imagen embebida (sin cámara — la OV2640 del Sense
// no responde a probe SCCB en mi hardware actual).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" const unsigned char g_mcunet_person_det[];
extern "C" const unsigned int g_mcunet_person_det_len;
extern "C" const float g_test_image_pd[];
extern "C" const int g_test_image_pd_w;
extern "C" const int g_test_image_pd_h;

namespace {
constexpr int kInputW = 160;
constexpr int kInputH = 128;
constexpr int kArenaBytes = 900 * 1024;

EXT_RAM_BSS_ATTR uint8_t tensor_arena[kArenaBytes];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
const char* TAG = "person_det";

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
}  // namespace

static void SetupModel() {
    if (g_mcunet_person_det_len < 100) {
        ESP_LOGE(TAG, "model_data.cc placeholder — bake the real .tflite first");
        vTaskDelay(portMAX_DELAY);
    }
    model = tflite::GetModel(g_mcunet_person_det);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "schema mismatch");
        vTaskDelay(portMAX_DELAY);
    }
    static tflite::MicroMutableOpResolver<16> resolver;
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
    resolver.AddResizeNearestNeighbor();
    resolver.AddLogistic();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaBytes);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (arena too small?)");
        vTaskDelay(portMAX_DELAY);
    }
    input = interpreter->input(0);
    ESP_LOGI(TAG, "model OK: in=[%d,%d,%d,%d] type=%d  arena=%d/%d  outputs=%d",
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3], input->type,
             interpreter->arena_used_bytes(), kArenaBytes,
             (int)interpreter->outputs_size());
    for (size_t i = 0; i < interpreter->outputs_size(); ++i) {
        const TfLiteTensor* o = interpreter->output(i);
        ESP_LOGI(TAG, "  out[%d]: [%d,%d,%d,%d] type=%d",
                 (int)i, o->dims->data[0], o->dims->data[1],
                 o->dims->data[2], o->dims->data[3], o->type);
    }
}

// Encuentra celda+anchor con mayor sigmoid(obj_logit)*sigmoid(class_logit) en una cabeza.
static float FindBestDetection(const TfLiteTensor* head, int* best_r, int* best_c, int* best_a) {
    const int gH = head->dims->data[1];
    const int gW = head->dims->data[2];
    const float* data = head->data.f;
    float best = -1e9f;
    *best_r = *best_c = *best_a = 0;
    for (int r = 0; r < gH; ++r) {
        for (int c = 0; c < gW; ++c) {
            for (int a = 0; a < 3; ++a) {
                const float obj = data[((r*gW + c)*18) + a*6 + 4];
                const float cls = data[((r*gW + c)*18) + a*6 + 5];
                float score = sigmoidf(obj) * sigmoidf(cls);
                if (score > best) {
                    best = score; *best_r = r; *best_c = c; *best_a = a;
                }
            }
        }
    }
    return best;
}

extern "C" void app_main() {
    ESP_LOGI(TAG, "MCUNet person-det demo (128x160 float32)");
    ESP_LOGI(TAG, "free heap internal: %d   PSRAM: %d",
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    SetupModel();

    int64_t lat_sum_us = 0;
    int frames = 0;
    while (true) {
        memcpy(input->data.f, g_test_image_pd, kInputW * kInputH * 3 * sizeof(float));

        int64_t t0 = esp_timer_get_time();
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        int64_t dt = esp_timer_get_time() - t0;
        lat_sum_us += dt;
        ++frames;

        int r, c, a;
        float s_lg = FindBestDetection(interpreter->output(0), &r, &c, &a);
        const int gH0 = interpreter->output(0)->dims->data[1];
        const int gW0 = interpreter->output(0)->dims->data[2];

        float s_md = FindBestDetection(interpreter->output(1), &r, &c, &a);
        const int gH1 = interpreter->output(1)->dims->data[1];
        const int gW1 = interpreter->output(1)->dims->data[2];

        float s_sm = FindBestDetection(interpreter->output(2), &r, &c, &a);
        const int gH2 = interpreter->output(2)->dims->data[1];
        const int gW2 = interpreter->output(2)->dims->data[2];

        const float best = (s_lg > s_md && s_lg > s_sm) ? s_lg :
                           (s_md > s_sm ? s_md : s_sm);
        const char* label = (best > 0.5f) ? "PERSON " : "no-pers";
        ESP_LOGI(TAG, "%s best=%.3f  heads=[%.3f@%dx%d, %.3f@%dx%d, %.3f@%dx%d]  inv=%lld us  avg=%lld us",
                 label, best,
                 s_lg, gH0, gW0,
                 s_md, gH1, gW1,
                 s_sm, gH2, gW2,
                 dt, lat_sum_us / frames);
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

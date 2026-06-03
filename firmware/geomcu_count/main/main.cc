// geomcu_count: MiCrowdNet (logits_int8) crowd counting sobre XIAO
// ESP32-S3 Sense.
//
// Modelo: microwd_paper_logits_int8_patch_128x160.tflite (158 KB).
// Input 128x160x3 int8, output 32x40x1 int8 LOGITS. El softplus final
// se aplica en fp32 en CPU host después del dequant para evitar la
// cuantización de EXP/LOG (rango dinámico amplio, colapsa int8).
//
// Estado: proof-of-concept single-patch. El firmware carga el modelo,
// corre 1 invoke sobre la imagen embebida y reporta latencia + count.
// Para llegar a inferencia full-image hay que iterar 64 patches con
// 8x8 tiling + halo 16; con la latencia actual de ~47 s/patch eso
// daría ~50 min/imagen, así que ese paso queda pendiente hasta
// resolver el bottleneck (DWConv kernels grandes fuera del fast-path
// de esp-nn). Ver firmware/geomcu_count/README.md.

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
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" const unsigned char g_microwd_paper_logits_int8_patch_128x160[];
extern "C" const unsigned int g_microwd_paper_logits_int8_patch_128x160_len;

extern "C" const int8_t g_test_patch[];
extern "C" const int g_test_patch_w;
extern "C" const int g_test_patch_h;

namespace {
// Patch del modelo: 128x160 input (8x8 tiling de 768x1024 + halo 16),
// 32x40 output (stride 4).
constexpr int kInputW = 160;
constexpr int kInputH = 128;
constexpr int kOutW = 40;
constexpr int kOutH = 32;
constexpr int kHaloIn = 16;        // px input (halo 16 de cada lado)
constexpr int kHaloOut = kHaloIn / 4;  // 4 px output
constexpr int kInteriorOutW = 32;  // 128/4
constexpr int kInteriorOutH = 24;  // 96/4

// Empírico: input 256x320 pidió 25.6 MB. Escala lineal con área → 128x160
// ~ 6.4 MB. Reservamos 7 MB y verificamos arena_used_bytes() en runtime.
constexpr size_t kArenaBytes = 7 * 1024 * 1024;

EXT_RAM_BSS_ATTR uint8_t tensor_arena[kArenaBytes];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

const char* TAG = "geomcu";
}  // namespace

static void SetupModel() {
    if (g_microwd_paper_logits_int8_patch_128x160_len < 1000) {
        ESP_LOGE(TAG, "model_data.cc tiene placeholder, bake la .tflite real "
                      "(host/geomcu/convert_to_c_array.sh)");
        while (true) vTaskDelay(portMAX_DELAY);
    }
    model = tflite::GetModel(g_microwd_paper_logits_int8_patch_128x160);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "schema mismatch: model=%lu lib=%d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        while (true) vTaskDelay(portMAX_DELAY);
    }

    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddAdd();
    resolver.AddConcatenation();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaBytes);
    interpreter = &static_interpreter;

    TfLiteStatus alloc = interpreter->AllocateTensors();
    if (alloc != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() FAILED, arena %u KB no alcanza",
                 (unsigned)(kArenaBytes / 1024));
        while (true) vTaskDelay(portMAX_DELAY);
    }
    input = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI(TAG, "model OK");
    ESP_LOGI(TAG, "  input  : %d dims, shape=[%d,%d,%d,%d] dtype=%d",
             input->dims->size,
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3],
             input->type);
    ESP_LOGI(TAG, "  output : shape=[%d,%d,%d,%d] dtype=%d",
             output->dims->data[0], output->dims->data[1],
             output->dims->data[2], output->dims->data[3],
             output->type);
    ESP_LOGI(TAG, "  arena  : used=%u / %u KB",
             (unsigned)(interpreter->arena_used_bytes() / 1024),
             (unsigned)(kArenaBytes / 1024));
    ESP_LOGI(TAG, "  in q   : scale=%f zp=%d",
             input->params.scale, (int)input->params.zero_point);
    ESP_LOGI(TAG, "  out q  : scale=%f zp=%d",
             output->params.scale, (int)output->params.zero_point);
}

// softplus(x) = log(1 + exp(x)), numéricamente estable.
static inline float Softplus(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

extern "C" void app_main() {
    ESP_LOGI(TAG, "geomcu_count: MiCrowdNet logits_int8 single-patch proof");
    ESP_LOGI(TAG, "input %dx%d -> output %dx%d (stride 4)",
             kInputW, kInputH, kOutW, kOutH);

    SetupModel();

    if (g_test_patch_w != kInputW || g_test_patch_h != kInputH) {
        ESP_LOGE(TAG, "test_patch shape %dx%d != esperado %dx%d",
                 g_test_patch_w, g_test_patch_h, kInputW, kInputH);
        while (true) vTaskDelay(portMAX_DELAY);
    }

    // Copia el patch al input tensor.
    std::memcpy(input->data.int8, g_test_patch,
                kInputW * kInputH * 3);

    int64_t t0 = esp_timer_get_time();
    TfLiteStatus inv = interpreter->Invoke();
    int64_t t1 = esp_timer_get_time();

    if (inv != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke FAILED");
        while (true) vTaskDelay(portMAX_DELAY);
    }
    ESP_LOGI(TAG, "Invoke OK, latency = %lld ms", (t1 - t0) / 1000);

    // Dequant + softplus + sum sobre el INTERIOR (sin halo).
    const float out_scale = output->params.scale;
    const int out_zp = output->params.zero_point;
    const int8_t* out_int = output->data.int8;

    float density_sum_interior = 0.0f;
    float density_sum_full = 0.0f;
    float max_density = 0.0f;
    for (int y = 0; y < kOutH; ++y) {
        for (int x = 0; x < kOutW; ++x) {
            const int8_t q = out_int[y * kOutW + x];
            const float logit = (q - out_zp) * out_scale;
            const float dens = Softplus(logit);
            density_sum_full += dens;
            if (dens > max_density) max_density = dens;
            const bool in_interior =
                (y >= kHaloOut && y < kHaloOut + kInteriorOutH) &&
                (x >= kHaloOut && x < kHaloOut + kInteriorOutW);
            if (in_interior) density_sum_interior += dens;
        }
    }

    ESP_LOGI(TAG, "density: sum_full=%.4f  sum_interior=%.4f  max=%.4f",
             density_sum_full, density_sum_interior, max_density);
    ESP_LOGI(TAG, "DONE. Compara con host: ver firmware/geomcu_count/README.md.");

    while (true) vTaskDelay(pdMS_TO_TICKS(60000));
}

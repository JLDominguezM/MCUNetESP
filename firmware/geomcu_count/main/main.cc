// geomcu_count: MiCrowdNet (logits_int8) crowd counting sobre XIAO
// ESP32-S3 Sense.
//
// Modelo: microwd_paper_logits_int8_patch_128x160.tflite (158 KB).
// Input 128x160x3 int8, output 32x40x1 int8 LOGITS. El softplus final
// se aplica en fp32 en CPU host después del dequant para evitar la
// cuantización de EXP/LOG (rango dinámico amplio, colapsa int8).
//
// Si la OV2640 inicializa: captura N frames RGB565 240x240, dumpea cada
// uno en base64 por serial, extrae el centro 160x128 RGB888, cuantiza,
// corre invoke y reporta count + latency.
// Si la cámara no responde: corre 1 invoke sobre la imagen estática
// embebida (g_test_patch) como sanity check.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_camera.h"

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
constexpr int kInputW = 160;
constexpr int kInputH = 128;
constexpr int kOutW = 40;
constexpr int kOutH = 32;
constexpr int kHaloOut = 4;
constexpr int kInteriorOutW = 32;
constexpr int kInteriorOutH = 24;

constexpr size_t kArenaBytes = 7 * 1024 * 1024;

// Camera capture: RGB565 240x240. Center crop a 160x128 al hacer el copy.
constexpr int kCamW = 240;
constexpr int kCamH = 240;

EXT_RAM_BSS_ATTR uint8_t tensor_arena[kArenaBytes];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

const char* TAG = "geomcu";
}  // namespace

static void SetupModel() {
    if (g_microwd_paper_logits_int8_patch_128x160_len < 1000) {
        ESP_LOGE(TAG, "model_data.cc placeholder, bake la .tflite real");
        while (true) vTaskDelay(portMAX_DELAY);
    }
    model = tflite::GetModel(g_microwd_paper_logits_int8_patch_128x160);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "schema mismatch");
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
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() FAILED");
        while (true) vTaskDelay(portMAX_DELAY);
    }
    input = interpreter->input(0);
    output = interpreter->output(0);
    ESP_LOGI(TAG, "model OK, arena used=%u / %u KB",
             (unsigned)(interpreter->arena_used_bytes() / 1024),
             (unsigned)(kArenaBytes / 1024));
    ESP_LOGI(TAG, "in q scale=%f zp=%d  out q scale=%f zp=%d",
             input->params.scale, (int)input->params.zero_point,
             output->params.scale, (int)output->params.zero_point);
}

static esp_err_t SetupCamera() {
    camera_config_t c = {};
    c.pin_pwdn = -1;
    c.pin_reset = -1;
    c.pin_xclk = 10;
    c.pin_sccb_sda = 40;
    c.pin_sccb_scl = 39;
    c.pin_d7 = 48; c.pin_d6 = 11; c.pin_d5 = 12; c.pin_d4 = 14;
    c.pin_d3 = 16; c.pin_d2 = 18; c.pin_d1 = 17; c.pin_d0 = 15;
    c.pin_vsync = 38;
    c.pin_href = 47;
    c.pin_pclk = 13;
    c.xclk_freq_hz = 20000000;
    c.ledc_timer = LEDC_TIMER_0;
    c.ledc_channel = LEDC_CHANNEL_0;
    c.pixel_format = PIXFORMAT_RGB565;
    c.frame_size = FRAMESIZE_240X240;
    c.jpeg_quality = 12;
    c.fb_count = 1;
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = CAMERA_GRAB_LATEST;
    c.sccb_i2c_port = 0;
    return esp_camera_init(&c);
}

static const char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void EmitBase64(const uint8_t* data, size_t len) {
    char line[80];
    size_t col = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i+1] << 8;
        if (i + 2 < len) v |= (uint32_t)data[i+2];
        line[col++] = kB64[(v >> 18) & 63];
        line[col++] = kB64[(v >> 12) & 63];
        line[col++] = (i + 1 < len) ? kB64[(v >> 6) & 63] : '=';
        line[col++] = (i + 2 < len) ? kB64[v & 63] : '=';
        if (col >= 76) {
            line[col] = 0;
            printf("FRAME_B64: %s\n", line);
            col = 0;
        }
    }
    if (col > 0) {
        line[col] = 0;
        printf("FRAME_B64: %s\n", line);
    }
}

// RGB565 big-endian -> RGB888 int8 (offset -128), recortando el centro
// 160x128 de un frame 240x240. Escribe directamente al input tensor.
static void CropAndQuantize(const uint8_t* fb_rgb565) {
    const int x0 = (kCamW - kInputW) / 2;  // 40
    const int y0 = (kCamH - kInputH) / 2;  // 56
    int8_t* dst = input->data.int8;
    for (int y = 0; y < kInputH; ++y) {
        const uint8_t* row = fb_rgb565 + 2 * ((y0 + y) * kCamW + x0);
        for (int x = 0; x < kInputW; ++x) {
            uint16_t pix = ((uint16_t)row[0] << 8) | row[1];
            row += 2;
            int r = ((pix >> 11) & 0x1F) << 3;
            int g = ((pix >> 5) & 0x3F) << 2;
            int b = (pix & 0x1F) << 3;
            *dst++ = (int8_t)(r - 128);
            *dst++ = (int8_t)(g - 128);
            *dst++ = (int8_t)(b - 128);
        }
    }
}

static inline float Softplus(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

static float RunInferenceAndSum() {
    int64_t t0 = esp_timer_get_time();
    TfLiteStatus inv = interpreter->Invoke();
    int64_t t1 = esp_timer_get_time();
    if (inv != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke FAILED");
        return -1.0f;
    }
    const float out_scale = output->params.scale;
    const int out_zp = output->params.zero_point;
    const int8_t* out_int = output->data.int8;
    float sum_full = 0.0f, sum_interior = 0.0f, max_d = 0.0f;
    for (int y = 0; y < kOutH; ++y) {
        for (int x = 0; x < kOutW; ++x) {
            float logit = (out_int[y * kOutW + x] - out_zp) * out_scale;
            float d = Softplus(logit);
            sum_full += d;
            if (d > max_d) max_d = d;
            if (y >= kHaloOut && y < kHaloOut + kInteriorOutH &&
                x >= kHaloOut && x < kHaloOut + kInteriorOutW) {
                sum_interior += d;
            }
        }
    }
    printf("RESULT: latency_ms=%lld sum_full=%.4f sum_interior=%.4f max=%.4f\n",
           (t1 - t0) / 1000, sum_full, sum_interior, max_d);
    return sum_full;
}

extern "C" void app_main() {
    ESP_LOGI(TAG, "geomcu_count live: input %dx%d crop centrado de cam %dx%d",
             kInputW, kInputH, kCamW, kCamH);
    SetupModel();

    esp_err_t cam_err = SetupCamera();
    bool have_cam = (cam_err == ESP_OK);
    if (!have_cam) {
        ESP_LOGW(TAG, "Camera no responde (%s). Modo estático con g_test_patch.",
                 esp_err_to_name(cam_err));
    } else {
        ESP_LOGI(TAG, "Camera OK: RGB565 %dx%d", kCamW, kCamH);
    }

    if (!have_cam) {
        std::memcpy(input->data.int8, g_test_patch, kInputW * kInputH * 3);
        RunInferenceAndSum();
        ESP_LOGI(TAG, "DONE static.");
        while (true) vTaskDelay(portMAX_DELAY);
    }

    const int N = 4;
    const int delay_ms = 2000;
    for (int n = 0; n < N; ++n) {
        camera_fb_t* fb = nullptr;
        for (int attempt = 0; attempt < 10; ++attempt) {
            vTaskDelay(pdMS_TO_TICKS(200));
            fb = esp_camera_fb_get();
            if (fb) break;
        }
        if (!fb) { ESP_LOGE(TAG, "fb_get NULL frame %d", n); continue; }
        ESP_LOGI(TAG, "frame %d/%d: %dx%d fmt=%d len=%zu",
                 n + 1, N, fb->width, fb->height, fb->format, fb->len);
        printf("===== FRAME_BEGIN fmt=%d w=%d h=%d len=%zu n=%d =====\n",
               fb->format, fb->width, fb->height, fb->len, n);
        EmitBase64(fb->buf, fb->len);
        printf("===== FRAME_END =====\n");
        CropAndQuantize(fb->buf);
        esp_camera_fb_return(fb);
        RunInferenceAndSum();
        if (n < N - 1) {
            ESP_LOGI(TAG, "wait %d ms para reposicionar", delay_ms);
            vTaskDelay(pdMS_TO_TICKS(delay_ms));
        }
    }
    ESP_LOGI(TAG, "DONE. Reset para capturar otra ronda.");
    while (true) vTaskDelay(portMAX_DELAY);
}

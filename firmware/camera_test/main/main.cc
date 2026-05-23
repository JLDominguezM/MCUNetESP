// Camera test: prueba varias configs hasta encontrar una que el OV2640 acepte.
// Cuando encuentra una, captura 1 frame JPEG y lo dumpea por serial en base64.
// Decoded en host por ../../host/decode_camera_frame.py.
//
// Nota: el chip ESP32-S3 a veces queda en estado raro entre flashes — si
// la cámara no responde, hacer un power cycle manual y reflash.
#include <cstdio>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_camera.h"

static const char* TAG = "cam_test";

static camera_config_t MakeConfig(int xclk_hz, pixformat_t fmt, framesize_t fs) {
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
    c.xclk_freq_hz = xclk_hz;
    c.ledc_timer = LEDC_TIMER_0;
    c.ledc_channel = LEDC_CHANNEL_0;
    c.pixel_format = fmt;
    c.frame_size = fs;
    c.jpeg_quality = 12;
    c.fb_count = 1;
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = CAMERA_GRAB_LATEST;
    c.sccb_i2c_port = 0;
    return c;
}

static const char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
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

extern "C" void app_main() {
    ESP_LOGI(TAG, "OV2640 probe sweep — XIAO ESP32-S3 Sense");

    struct Cfg { int xclk; pixformat_t fmt; framesize_t fs; const char* name; };
    Cfg configs[] = {
        {20000000, PIXFORMAT_RGB565,  FRAMESIZE_240X240, "20MHz RGB565 240x240"},
        {10000000, PIXFORMAT_RGB565,  FRAMESIZE_240X240, "10MHz RGB565 240x240"},
        {20000000, PIXFORMAT_JPEG,    FRAMESIZE_QVGA,    "20MHz JPEG QVGA"},
        {10000000, PIXFORMAT_JPEG,    FRAMESIZE_QVGA,    "10MHz JPEG QVGA"},
    };

    bool ok = false;
    Cfg working = configs[0];
    for (const auto& c : configs) {
        ESP_LOGI(TAG, "trying: %s", c.name);
        camera_config_t cc = MakeConfig(c.xclk, c.fmt, c.fs);
        esp_err_t e = esp_camera_init(&cc);
        if (e == ESP_OK) {
            ESP_LOGI(TAG, ">>> SUCCESS with %s", c.name);
            ok = true; working = c;
            break;
        } else {
            ESP_LOGW(TAG, "  fail: %s (%s)", c.name, esp_err_to_name(e));
            esp_camera_deinit();
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }

    if (!ok) {
        ESP_LOGE(TAG, "ALL CONFIGS FAILED.");
        ESP_LOGE(TAG, "El chip puede estar en estado atascado — power cycle físico y reflash.");
        while (true) vTaskDelay(portMAX_DELAY);
    }

    // Reintentos: el sensor necesita unos frames para estabilizar AGC/AEC.
    camera_fb_t* fb = nullptr;
    for (int attempt = 1; attempt <= 15; ++attempt) {
        vTaskDelay(pdMS_TO_TICKS(200));
        fb = esp_camera_fb_get();
        if (fb) {
            ESP_LOGI(TAG, "fb_get OK on attempt %d", attempt);
            break;
        }
        ESP_LOGW(TAG, "fb_get NULL (attempt %d/15) — retrying ...", attempt);
        if (fb) { esp_camera_fb_return(fb); fb = nullptr; }
    }
    if (!fb) {
        ESP_LOGE(TAG, "fb_get NULL after 15 attempts — DVP/DMA not producing frames");
        while (true) vTaskDelay(portMAX_DELAY);
    }

    ESP_LOGI(TAG, "captured frame: %dx%d  fmt=%d  len=%zu bytes",
             fb->width, fb->height, fb->format, fb->len);
    ESP_LOGI(TAG, "===== FRAME_BEGIN fmt=%d w=%d h=%d len=%zu =====",
             fb->format, fb->width, fb->height, fb->len);
    EmitBase64(fb->buf, fb->len);
    ESP_LOGI(TAG, "===== FRAME_END =====");

    esp_camera_fb_return(fb);
    ESP_LOGI(TAG, "done. Reset to capture again.");
    while (true) vTaskDelay(portMAX_DELAY);
}

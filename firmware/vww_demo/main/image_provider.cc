#include "image_provider.h"

#include "esp_camera.h"
#include "esp_log.h"

static const char* TAG = "image_provider";

// Pinout XIAO ESP32-S3 Sense (ver docs/camera_pinout_xiao.md).
// mcunet-vww2 espera 144x144 RGB int8 (verificado con eval_tflite.py en host).
// Capturamos a 240x240 RGB565 y recortamos el centro 144x144.
static camera_config_t kCameraConfig = {
    .pin_pwdn = -1,
    .pin_reset = -1,
    .pin_xclk = 10,
    .pin_sccb_sda = 40,
    .pin_sccb_scl = 39,
    .pin_d7 = 48,
    .pin_d6 = 11,
    .pin_d5 = 12,
    .pin_d4 = 14,
    .pin_d3 = 16,
    .pin_d2 = 18,
    .pin_d1 = 17,
    .pin_d0 = 15,
    .pin_vsync = 38,
    .pin_href = 47,
    .pin_pclk = 13,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_RGB565,
    .frame_size = FRAMESIZE_240X240,
    .jpeg_quality = 12,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_LATEST,
};

esp_err_t ImageProviderInit() {
    esp_err_t err = esp_camera_init(&kCameraConfig);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init failed: %s", esp_err_to_name(err));
    }
    return err;
}

// Recorta el centro `width`x`height` del frame RGB565 y lo convierte a int8 RGB888
// con zero_point = -128 (cuantización simétrica TFLite int8).
esp_err_t GetImage(int8_t* out, int width, int height) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "esp_camera_fb_get returned NULL");
        return ESP_FAIL;
    }
    const int src_w = fb->width;
    const int src_h = fb->height;
    const int off_x = (src_w - width) / 2;
    const int off_y = (src_h - height) / 2;
    if (off_x < 0 || off_y < 0) {
        ESP_LOGE(TAG, "frame %dx%d smaller than crop %dx%d", src_w, src_h, width, height);
        esp_camera_fb_return(fb);
        return ESP_ERR_INVALID_SIZE;
    }
    const uint16_t* src = reinterpret_cast<const uint16_t*>(fb->buf);
    for (int y = 0; y < height; ++y) {
        const uint16_t* row = src + (y + off_y) * src_w + off_x;
        int8_t* dst = out + y * width * 3;
        for (int x = 0; x < width; ++x) {
            uint16_t p = row[x];
            int r = ((p >> 11) & 0x1F) << 3;
            int g = ((p >> 5)  & 0x3F) << 2;
            int b = ( p        & 0x1F) << 3;
            dst[3*x + 0] = static_cast<int8_t>(r - 128);
            dst[3*x + 1] = static_cast<int8_t>(g - 128);
            dst[3*x + 2] = static_cast<int8_t>(b - 128);
        }
    }
    esp_camera_fb_return(fb);
    return ESP_OK;
}

// Lee el chip ID del sensor en 0x30 para confirmar exactamente qué modelo es.
//
// Procedimiento OV2640 según hoja de datos:
//   1. Escribir reg 0xFF = 0x01 (selecciona "sensor bank" — registros del sensor)
//   2. Leer reg 0x0A → PID (Product ID, MSB)
//   3. Leer reg 0x0B → VER (Version, LSB)
//   Esperado: PID=0x26, VER=0x42 → "0x2642" = OV2640
//
// También probamos OV3660 (PID en 0x300A/0x300B → 0x3660) y OV5640 (0x300A/0x300B → 0x5640).
#include <cstdio>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/i2c_master.h"
#include "driver/ledc.h"

static const char* TAG = "chipid";

#define PIN_XCLK 10
#define PIN_SDA  40
#define PIN_SCL  39
#define XCLK_HZ  20000000

#define SCCB_ADDR 0x30  // 7-bit address of OV-family sensors

static i2c_master_dev_handle_t dev = nullptr;

static void start_xclk() {
    ledc_timer_config_t timer = {};
    timer.speed_mode = LEDC_LOW_SPEED_MODE;
    timer.timer_num = LEDC_TIMER_0;
    timer.duty_resolution = LEDC_TIMER_1_BIT;
    timer.freq_hz = XCLK_HZ;
    timer.clk_cfg = LEDC_AUTO_CLK;
    ESP_ERROR_CHECK(ledc_timer_config(&timer));
    ledc_channel_config_t ch = {};
    ch.gpio_num = PIN_XCLK;
    ch.speed_mode = LEDC_LOW_SPEED_MODE;
    ch.channel = LEDC_CHANNEL_0;
    ch.timer_sel = LEDC_TIMER_0;
    ch.duty = 1;
    ESP_ERROR_CHECK(ledc_channel_config(&ch));
}

// SCCB write 1 byte to a 1-byte register address.
static esp_err_t sccb_write1(uint8_t reg, uint8_t val) {
    uint8_t buf[2] = {reg, val};
    return i2c_master_transmit(dev, buf, 2, 100);
}

// SCCB read 1 byte from a 1-byte register address (OV2640 style — split txns).
static esp_err_t sccb_read1(uint8_t reg, uint8_t* out) {
    esp_err_t e = i2c_master_transmit(dev, &reg, 1, 100);
    if (e != ESP_OK) return e;
    return i2c_master_receive(dev, out, 1, 100);
}

// SCCB read 1 byte from a 2-byte register address (OV3660/OV5640 style).
static esp_err_t sccb_read16(uint16_t reg, uint8_t* out) {
    uint8_t r[2] = {(uint8_t)(reg >> 8), (uint8_t)(reg & 0xFF)};
    esp_err_t e = i2c_master_transmit(dev, r, 2, 100);
    if (e != ESP_OK) return e;
    return i2c_master_receive(dev, out, 1, 100);
}

extern "C" void app_main() {
    ESP_LOGI(TAG, "OV-family chip ID reader on 0x30");
    vTaskDelay(pdMS_TO_TICKS(100));

    start_xclk();
    ESP_LOGI(TAG, "XCLK %d MHz on GPIO%d; settling 500 ms ...", XCLK_HZ/1000000, PIN_XCLK);
    vTaskDelay(pdMS_TO_TICKS(500));

    i2c_master_bus_config_t bus_config = {};
    bus_config.i2c_port = -1;
    bus_config.sda_io_num = (gpio_num_t)PIN_SDA;
    bus_config.scl_io_num = (gpio_num_t)PIN_SCL;
    bus_config.clk_source = I2C_CLK_SRC_DEFAULT;
    bus_config.glitch_ignore_cnt = 7;
    bus_config.flags.enable_internal_pullup = true;
    i2c_master_bus_handle_t bus = nullptr;
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_config, &bus));

    i2c_device_config_t dev_config = {};
    dev_config.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    dev_config.device_address = SCCB_ADDR;
    dev_config.scl_speed_hz = 100000;
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus, &dev_config, &dev));
    ESP_LOGI(TAG, "I2C device 0x%02X registered (100 kHz)", SCCB_ADDR);

    // ─── Try OV2640 protocol (8-bit register addresses, banked) ────────────────
    uint8_t pid_2640 = 0, ver_2640 = 0;
    esp_err_t e1 = sccb_write1(0xFF, 0x01);  // sensor bank
    vTaskDelay(pdMS_TO_TICKS(10));
    esp_err_t e2 = sccb_read1(0x0A, &pid_2640);
    esp_err_t e3 = sccb_read1(0x0B, &ver_2640);
    ESP_LOGI(TAG, "OV2640 protocol  → write 0xFF=0x01: %s",
             e1==ESP_OK?"OK":esp_err_to_name(e1));
    ESP_LOGI(TAG, "                 → read  0x0A (PID): %s value=0x%02X",
             e2==ESP_OK?"OK":esp_err_to_name(e2), pid_2640);
    ESP_LOGI(TAG, "                 → read  0x0B (VER): %s value=0x%02X",
             e3==ESP_OK?"OK":esp_err_to_name(e3), ver_2640);

    if (e2==ESP_OK && e3==ESP_OK) {
        uint16_t id = ((uint16_t)pid_2640 << 8) | ver_2640;
        ESP_LOGI(TAG, "  Combined ID = 0x%04X", id);
        switch (id) {
            case 0x2640: case 0x2641: case 0x2642:
                ESP_LOGI(TAG, "  >>> Es un OV2640 (variante 0x%04X) <<<", id); break;
            case 0x7670: ESP_LOGI(TAG, "  >>> OV7670 <<<"); break;
            case 0x7725: ESP_LOGI(TAG, "  >>> OV7725 <<<"); break;
            default:
                ESP_LOGW(TAG, "  ID desconocido en protocolo OV2640.");
        }
    }

    // ─── Try OV3660/5640 protocol (16-bit register addresses) ──────────────────
    uint8_t pid_hi = 0, pid_lo = 0;
    esp_err_t e4 = sccb_read16(0x300A, &pid_hi);
    esp_err_t e5 = sccb_read16(0x300B, &pid_lo);
    ESP_LOGI(TAG, "OV3660/5640 protocol → 0x300A: %s = 0x%02X, 0x300B: %s = 0x%02X",
             e4==ESP_OK?"OK":esp_err_to_name(e4), pid_hi,
             e5==ESP_OK?"OK":esp_err_to_name(e5), pid_lo);
    if (e4==ESP_OK && e5==ESP_OK) {
        uint16_t id = ((uint16_t)pid_hi << 8) | pid_lo;
        ESP_LOGI(TAG, "  Combined ID = 0x%04X", id);
        switch (id) {
            case 0x3660: ESP_LOGI(TAG, "  >>> OV3660 <<<"); break;
            case 0x5640: ESP_LOGI(TAG, "  >>> OV5640 <<<"); break;
            case 0x5648: ESP_LOGI(TAG, "  >>> OV5648 <<<"); break;
            default: ESP_LOGW(TAG, "  ID desconocido en protocolo 16-bit.");
        }
    }

    ESP_LOGI(TAG, "Done.");
    while (true) vTaskDelay(portMAX_DELAY);
}

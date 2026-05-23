#pragma once
#include <cstdint>
#include "esp_err.h"

// Inicializa el driver de cámara OV2640 a 96x96 grayscale.
// Llamar una sola vez en app_main antes del primer GetImage.
esp_err_t ImageProviderInit();

// Captura un frame y lo deja en `out` como int8 [-128, 127] cuantizado.
// `out` debe apuntar a un buffer de al menos 96*96 bytes.
// Devuelve ESP_OK en éxito.
esp_err_t GetImage(int8_t* out, int width, int height);

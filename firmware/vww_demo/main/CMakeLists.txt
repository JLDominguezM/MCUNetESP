idf_component_register(
    SRCS
        "main.cc"
        "model_data.cc"
        "image_provider.cc"
        "test_image.cc"
    INCLUDE_DIRS "."
    REQUIRES
        esp_timer
        spi_flash
)

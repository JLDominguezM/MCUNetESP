# Top-level CMakeLists.txt para geomcu_count (ESP-IDF v5.2+).
# Despliega microwd_paper_logits_int8.tflite en XIAO ESP32-S3 Sense via
# inferencia patch-based (MCUNetV2 style).
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(geomcu_count)

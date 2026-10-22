# examples/esp32-cam/firmware/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32cam_demo)

# Selección de placa: idf.py -DBOARD=<esp32|esp32cam|esp32s3|esp32s3cam>
if(NOT DEFINED BOARD)
    set(BOARD "esp32cam")
endif()
set(VALID_BOARDS esp32 esp32cam esp32s3 esp32s3cam)
if(NOT BOARD IN_LIST VALID_BOARDS)
    message(FATAL_ERROR "BOARD='${BOARD}' inválido. Válidos: ${VALID_BOARDS}")
endif()
string(TOUPPER "${BOARD}" BOARD_UPPER)
idf_build_set_property(COMPILE_DEFINITIONS "-DBOARD_${BOARD_UPPER}=1" APPEND)

# IDF 5.3 compat: MALLOC_CAP_SIMD was introduced in IDF 5.4.
# Inject a no-op define so ESP-DL 3.3.x builds cleanly on IDF 5.3.
# This must come AFTER project() so idf_build_set_property is available.
idf_build_set_property(COMPILE_DEFINITIONS "-DMALLOC_CAP_SIMD=0" APPEND)

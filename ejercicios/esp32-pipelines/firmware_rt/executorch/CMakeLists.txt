# firmware_rt/executorch — self-test ExecuTorch portable (float) en ESP32-S3.
#
# Requiere el runtime ExecuTorch cross-compilado en
#   third_party/executorch/cmake-out-esp/  (ver README.md)
cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../common")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(rt_executorch)

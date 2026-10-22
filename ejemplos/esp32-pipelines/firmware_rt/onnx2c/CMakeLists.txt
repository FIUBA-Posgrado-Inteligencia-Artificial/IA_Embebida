cmake_minimum_required(VERSION 3.16)

# Componente compartido selftest_runner (firmware_rt/common/)
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../common")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(onnx2c_rt)

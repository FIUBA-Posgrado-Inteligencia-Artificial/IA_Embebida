cmake_minimum_required(VERSION 3.16)
project(esp32cam_host_tests LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

enable_testing()

add_library(unity STATIC unity/unity.c)
target_include_directories(unity PUBLIC unity)

add_library(preprocess STATIC ../../main/preprocess.c)
target_include_directories(preprocess PUBLIC ../../main)
target_compile_options(preprocess PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_preprocess test_preprocess.c)
target_link_libraries(test_preprocess PRIVATE preprocess unity m)

add_test(NAME preprocess COMMAND test_preprocess)

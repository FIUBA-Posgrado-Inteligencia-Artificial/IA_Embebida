idf_component_register(SRCS "main.cpp"
                       INCLUDE_DIRS "."
                       REQUIRES common esp_timer heap)

# Runtime ExecuTorch pre-cross-compilado (ver README.md).
set(ET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/executorch")
set(ET_BUILD_DIR "${ET_ROOT}/cmake-out-esp"
    CACHE PATH "Install dir del build Xtensa de ExecuTorch")

# GLOBAL: project.cmake resuelve los links a nivel proyecto, pero
# find_package corre en el scope del componente (mismo truco que
# examples/espressif/executor_runner de executorch).
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)
find_package(executorch REQUIRED HINTS "${ET_BUILD_DIR}/lib/cmake/ExecuTorch")

target_include_directories(${COMPONENT_LIB} PRIVATE
    "${ET_ROOT}/.."                                 # #include <executorch/...>
    "${ET_ROOT}/runtime/core/portable_type/c10")
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    C10_USING_CUSTOM_GENERATED_MACROS)

# OJO: NO agregar -Wl,--whole-archive manual acá. El target importado
# portable_ops_lib ya lo trae vía INTERFACE_LINK_OPTIONS (con TARGET_FILE);
# duplicarlo mete los ctors de registración DOS veces en el ELF →
# "Re-registering ..." + LoadProhibited en el log antes del PAL init.
target_link_libraries(${COMPONENT_LIB} PUBLIC
    executorch
    portable_ops_lib
    portable_kernels)

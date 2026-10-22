idf_component_register(SRCS "main.c" "model.c"
                       INCLUDE_DIRS "."
                       REQUIRES common esp_timer)

# model.c es código generado (onnx2c): no aplicar -Werror ni warnings extra.
set_source_files_properties(model.c PROPERTIES COMPILE_OPTIONS "-w;-O2")

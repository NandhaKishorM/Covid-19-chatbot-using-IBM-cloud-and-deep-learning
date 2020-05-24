cmake_minimum_required(VERSION 3.5)

set(RN_NOISE_DEMO_SRC
        rnnoise_demo.c)

set(RN_NOISE_DEMO_SIMPLE_SRC
        rnnoise_demo_simple.c)

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")
include_directories("../include")
link_directories("${PROJECT_BINARY_DIR}/lib")
add_executable(rnnoise_demo ${RN_NOISE_DEMO_SRC})
target_link_libraries(rnnoise_demo -lrnnoise -lm -lrt)
add_executable(rnnoise_demo_simple ${RN_NOISE_DEMO_SIMPLE_SRC})
target_link_libraries(rnnoise_demo_simple -lrnnoise -lm -lrt)

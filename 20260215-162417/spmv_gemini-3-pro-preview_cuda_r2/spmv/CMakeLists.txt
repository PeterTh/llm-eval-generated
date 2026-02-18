cmake_minimum_required(VERSION 3.10)

project(spmv_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA manually if needed or just use nvcc
find_program(NVCC_EXECUTABLE nvcc PATHS /usr/local/cuda-12.6/bin)

if(NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

# Define the source files
set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/spmv.cpp)
set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/spmv)

# Custom command to compile with nvcc
add_custom_command(
    OUTPUT ${OUTPUT_FILE}
    COMMAND ${NVCC_EXECUTABLE} -x cu -O3 -std=c++17
            -I${CMAKE_CURRENT_SOURCE_DIR}/../common
            ${SOURCE_FILE} -o ${OUTPUT_FILE}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Compiling ${SOURCE_FILE} with nvcc"
)

add_custom_target(spmv_target ALL DEPENDS ${OUTPUT_FILE})

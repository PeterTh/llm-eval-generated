cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Rename source file to .cu for CUDA compilation
set(SOURCE_FILE qtclustering.cpp)

# Set CUDA architectures - support common GPUs
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# Set the source as CUDA
set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering ${SOURCE_FILE})

# CUDA-specific compiler options
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --extended-lambda
        -lineinfo
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
        -Wall
        -Wextra
    >
)

# Link CUDA runtime and math library
target_link_libraries(qtclustering cudart m)


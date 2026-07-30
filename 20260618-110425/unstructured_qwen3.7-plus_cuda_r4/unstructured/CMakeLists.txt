cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture for RTX 3090 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(unstructured unstructured.cpp)

# Treat the .cpp file as CUDA source
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for CUDA
target_compile_options(unstructured PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>)

# Link CUDA runtime and math library
target_link_libraries(unstructured cudart m)

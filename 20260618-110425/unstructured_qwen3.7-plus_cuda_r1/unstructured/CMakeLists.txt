cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Treat .cpp files as CUDA source files
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations for CUDA compilation
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -Xcompiler -march=native,-Wall,-Wextra>
)

# Link CUDA runtime and math libraries
target_link_libraries(unstructured cudart m)

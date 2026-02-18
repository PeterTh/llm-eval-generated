cmake_minimum_required(VERSION 3.10)

# Set CUDA paths before project
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)

project(floydwarshall_benchmark LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86 89 90)

add_executable(floydwarshall floydwarshall.cu)

set_target_properties(floydwarshall PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

# Link math library
target_link_libraries(floydwarshall m)

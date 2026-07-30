cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures for RTX 3090 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES "86")

add_executable(matmul matmul.cu)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Use fast math for better GPU performance
set_property(TARGET matmul PROPERTY CUDA_ARCHITECTURES "86")
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)

# Link math library
target_link_libraries(matmul m)

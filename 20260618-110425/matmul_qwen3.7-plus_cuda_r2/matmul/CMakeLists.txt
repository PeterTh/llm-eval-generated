cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(matmul matmul.cu)

# Enable optimizations for C++
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Enable optimizations for CUDA
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    --expt-relaxed-constexpr
>)

# Link math library and CUDA runtime
target_link_libraries(matmul m cudart)

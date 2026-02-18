cmake_minimum_required(VERSION 3.18)

# Set CUDA architectures before project() - use 86 for compute capability 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cu)

# Enable optimizations for both host and device code
target_compile_options(spmv PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)

# Link math library
target_link_libraries(spmv m)

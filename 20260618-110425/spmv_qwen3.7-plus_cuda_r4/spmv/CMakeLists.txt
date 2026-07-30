cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Set CUDA architecture (adjust based on your GPU)
# Common values: sm_60 (Pascal), sm_70 (Volta), sm_75 (Turing), sm_80 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 70)

add_executable(spmv spmv.cu)

# Enable optimizations for both C++ and CUDA
target_compile_options(spmv PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)

# Link CUDA runtime and math library
target_link_libraries(spmv cudart m)

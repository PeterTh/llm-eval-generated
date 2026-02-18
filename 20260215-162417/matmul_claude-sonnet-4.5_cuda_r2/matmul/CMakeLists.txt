cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cu)

# Set CUDA architecture - use native if available, otherwise common architectures
set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Try to detect CUDA architecture, but provide fallback
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
endif()

# Enable optimizations
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Link math library and CUDA runtime
target_link_libraries(matmul m cudart)

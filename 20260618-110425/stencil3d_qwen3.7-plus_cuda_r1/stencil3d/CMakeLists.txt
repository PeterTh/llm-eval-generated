cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for RTX 3090 (SM 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(stencil3d stencil3d.cu)

# Enable optimizations
target_compile_options(stencil3d PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

# Link math library and CUDA runtime
target_link_libraries(stencil3d m)

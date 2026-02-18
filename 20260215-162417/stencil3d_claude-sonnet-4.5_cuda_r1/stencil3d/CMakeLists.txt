cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cu)

# Set CUDA architectures (if not set, use common values)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
endif()

set_target_properties(stencil3d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations for both host and device code
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)

# Link math library
target_link_libraries(stencil3d m)


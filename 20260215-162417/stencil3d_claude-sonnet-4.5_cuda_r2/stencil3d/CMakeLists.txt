cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures before adding executable
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)
endif()

# Rename source file to .cu for CUDA compilation
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

set_target_properties(stencil3d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-Wall>
)

# Link math library
target_link_libraries(stencil3d m)

cmake_minimum_required(VERSION 3.18)

# Explicitly set architectures (avoid auto-detection failures in headless environments)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# Compile this translation unit with NVCC (contains CUDA kernels)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3>
)

# CUDA runtime + math library
find_package(CUDAToolkit REQUIRED)
target_link_libraries(stencil3d PRIVATE CUDA::cudart m)

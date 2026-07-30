cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required parallelization packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA architecture targets (Volta+ for broad compatibility, Ampere+ for best perf)
set(CMAKE_CUDA_ARCHITECTURES "70;80;86;90" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(stencil3d stencil3d.cpp)

# Compile as CUDA source so nvcc handles kernel syntax
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (for CUDA, flags are forwarded to host compiler via -Xcompiler)
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -O3;-Xcompiler -march=native;-Xcompiler -Wall;-Xcompiler -Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)

# OpenMP: pass -fopenmp to the host compiler through nvcc for CUDA files
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

# Link parallel libraries
target_link_libraries(stencil3d PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

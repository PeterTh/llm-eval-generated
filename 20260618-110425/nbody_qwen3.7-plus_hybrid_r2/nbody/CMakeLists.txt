cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Treat nbody.cpp as CUDA source (it contains __global__ kernels)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations for host code
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options "-O3 -march=native">
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link libraries
target_link_libraries(nbody
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

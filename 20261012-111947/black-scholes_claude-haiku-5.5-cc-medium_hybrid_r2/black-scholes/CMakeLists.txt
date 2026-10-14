cmake_minimum_required(VERSION 3.23)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Hybrid parallelization: MPI across ranks, OpenMP threads driving GPUs, CUDA kernels
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Build device code for every major GPU architecture, so the binary runs on any cluster GPU
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

# The benchmark source contains CUDA kernels, so it is compiled by nvcc
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations; host-side OpenMP flags are forwarded through nvcc
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=${OpenMP_CXX_FLAGS}>
)
target_link_options(black_scholes PRIVATE ${OpenMP_CXX_FLAGS})

target_link_libraries(black_scholes PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    m
)

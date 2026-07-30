cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Compile .cpp with CUDA compiler to support __global__ kernels
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -Wall -Wextra)

# Architecture-specific flags
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>)
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

# OpenMP must be passed through nvcc's host compiler when compiled as CUDA
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>)
target_link_options(black_scholes PRIVATE $<$<LINK_LANGUAGE:CUDA>:-fopenmp>)

# Link libraries
target_link_libraries(black_scholes
    PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

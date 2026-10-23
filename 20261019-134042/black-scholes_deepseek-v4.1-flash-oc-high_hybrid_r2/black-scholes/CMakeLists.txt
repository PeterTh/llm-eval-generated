cmake_minimum_required(VERSION 3.24)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The benchmark is a single translation unit containing CUDA kernels as well as
# the MPI/OpenMP host code, so it must be compiled by nvcc.
add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for both the device code and the host compiler.
target_compile_options(black_scholes PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp -Xcompiler=-march=native
                            -Xcompiler=-Wall -Xcompiler=-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp -Wall -Wextra>)

# MPI and OpenMP provide the inter-node and intra-node host parallelism; the
# CUDA runtime is linked automatically by CMake for the CUDA language.
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

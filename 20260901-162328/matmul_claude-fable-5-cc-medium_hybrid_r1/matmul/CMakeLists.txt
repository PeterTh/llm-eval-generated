cmake_minimum_required(VERSION 3.18)

# CUDA 12.0 supports host compilers up to GCC 12
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/g++-12)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
endif()

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# The single source file contains CUDA kernels; compile it with nvcc
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host-side flags passed through nvcc)
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp -Wall -Wextra>
)

# Link math library, MPI, and OpenMP
target_link_libraries(matmul m MPI::MPI_CXX OpenMP::OpenMP_CXX)

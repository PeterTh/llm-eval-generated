cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Hybrid MPI + OpenMP + CUDA build: all dependencies are mandatory
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
  set(CMAKE_CUDA_STANDARD 20)
else()
  set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
  endif()
endif()

# The source contains CUDA kernels, so it is compiled by nvcc
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Optimizations and OpenMP are forwarded to the host compiler through nvcc
target_compile_options(matmul PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra,-fopenmp)
# Link through the host compiler driver directly when CMake does so, otherwise through nvcc
if(CMAKE_CUDA_HOST_LINK_LAUNCHER)
  target_link_options(matmul PRIVATE -fopenmp)
else()
  target_link_options(matmul PRIVATE -Xcompiler=-fopenmp)
endif()

# MPI, CUDA runtime, and math library
target_link_libraries(matmul PRIVATE MPI::MPI_CXX CUDA::cudart m)

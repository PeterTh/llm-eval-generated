cmake_minimum_required(VERSION 3.18)

# Ensure nvcc is discoverable in minimal environments
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "" FORCE)
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to a broad set of common datacenter GPU architectures when not specified.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(spmv PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xcompiler=-O3;-Xcompiler=-march=native;-Xcompiler=-fopenmp;-Xcompiler=-Wall;-Xcompiler=-Wextra>
)

target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

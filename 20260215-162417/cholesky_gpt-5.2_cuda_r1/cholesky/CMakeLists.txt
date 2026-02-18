cmake_minimum_required(VERSION 3.18)

# Ensure we can find nvcc even when it's not in PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build tuned for the local GPU by default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(cholesky PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

target_link_libraries(cholesky PRIVATE m CUDA::cudart CUDA::cublas CUDA::cusolver)

cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Required parallel runtimes: MPI (inter-node), OpenMP (host threads), CUDA (GPU)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Default GPU architectures (override with -DCMAKE_CUDA_ARCHITECTURES=...)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
endif()

add_executable(unstructured unstructured.cpp)

# The single source file contains device code, so it is compiled by nvcc.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Host compiler flags are forwarded through nvcc
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-Xcompiler=-march=native;-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-fopenmp>
)
target_link_libraries(unstructured PRIVATE
  MPI::MPI_CXX
  CUDA::cudart
  $<LINK_ONLY:OpenMP::OpenMP_CXX>
  m
)

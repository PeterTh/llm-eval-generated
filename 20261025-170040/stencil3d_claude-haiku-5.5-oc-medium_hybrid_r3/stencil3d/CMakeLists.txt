cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The single source file contains CUDA kernels, so it is compiled by nvcc.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations (host code via -Xcompiler for nvcc)
target_compile_options(stencil3d PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-fopenmp>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native,-Wall,-Wextra,-fopenmp>)

target_link_options(stencil3d PRIVATE -fopenmp)

target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX CUDA::cudart m)

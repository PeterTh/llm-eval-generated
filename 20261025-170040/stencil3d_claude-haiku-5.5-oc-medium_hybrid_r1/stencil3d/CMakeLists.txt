cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
  endif()
endif()

# stencil3d.cpp contains both host code (MPI, OpenMP) and CUDA kernels,
# so it is compiled by nvcc with host-side OpenMP/optimization flags forwarded.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>)

# MPI, OpenMP, CUDA runtime, and math library (OpenMP link flag comes from the imported target)
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

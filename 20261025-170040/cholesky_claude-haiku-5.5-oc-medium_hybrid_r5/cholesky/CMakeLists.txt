cmake_minimum_required(VERSION 3.24)

# Default GPU targets (each includes PTX for forward compatibility). Must be set before
# project() so CMake's CUDA module does not substitute the compiler default. Override with
# -DCMAKE_CUDA_ARCHITECTURES=<list> for a specific GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89)
endif()

project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# MPI for distribution across ranks, CUDA for the GPU kernels, OpenMP for host threading.
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The program contains CUDA kernels, so its single source file is compiled by nvcc.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Host compiler flags are forwarded through nvcc. -fopenmp is required for both
# compilation (OpenMP pragmas) and linking.
target_compile_options(cholesky PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

target_link_options(cholesky PRIVATE -fopenmp)

target_link_libraries(cholesky PRIVATE
  MPI::MPI_CXX
  CUDA::cudart
  m
)

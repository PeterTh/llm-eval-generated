cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# The source contains CUDA kernels; compile it with nvcc
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host-compiler flags routed through nvcc)
target_compile_options(floydwarshall PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

target_link_libraries(floydwarshall PRIVATE
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CUDA::cudart
  m
)

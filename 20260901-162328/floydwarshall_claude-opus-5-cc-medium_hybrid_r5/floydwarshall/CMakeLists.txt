cmake_minimum_required(VERSION 3.24)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The benchmark source contains CUDA kernels and is compiled by nvcc.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)

target_link_libraries(floydwarshall
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CUDA::cudart
  m)

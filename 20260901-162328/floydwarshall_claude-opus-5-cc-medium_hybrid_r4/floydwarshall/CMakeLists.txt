cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The single source file contains CUDA kernels and is compiled by nvcc
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations (host compiler flags are forwarded through nvcc)
target_compile_options(floydwarshall PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xptxas=-O3;-Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
)

target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

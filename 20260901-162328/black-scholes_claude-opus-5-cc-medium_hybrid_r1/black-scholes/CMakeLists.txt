cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(black_scholes black_scholes.cpp)

# The single translation unit contains CUDA kernels; compile it with nvcc.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Enable optimizations (host flags are forwarded through nvcc)
target_compile_options(black_scholes PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)

target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

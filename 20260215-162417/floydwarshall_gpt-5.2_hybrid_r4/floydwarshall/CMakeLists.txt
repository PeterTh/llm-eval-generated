cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA compilation is required; keep this conservative for broad NVCC compatibility.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build fatbins for common accelerator clusters (V100/A100/H100 + Ampere consumer).
set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Compile the existing .cpp as CUDA (no new files allowed).
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host + device) and OpenMP for host code compiled via NVCC.
target_compile_options(floydwarshall PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-Xcompiler=-O3,-march=native,-fopenmp>
)

target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

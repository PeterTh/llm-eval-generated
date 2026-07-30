cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Compile matmul.cpp as CUDA to support kernels in .cpp source
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# CUDA architecture targets (Ampere+ for broad accelerator cluster compatibility)
set(CMAKE_CUDA_ARCHITECTURES "80;90")

target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

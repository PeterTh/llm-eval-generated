cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Enable SIMD without an OpenMP runtime or threads, and preserve dot-product
# accumulation order and separate multiplication/addition rounding.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -fopenmp-simd -Wall -Wextra)

# MPI is required even for a single-process invocation.
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

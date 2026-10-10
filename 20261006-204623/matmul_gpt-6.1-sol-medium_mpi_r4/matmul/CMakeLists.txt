cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Enable optimizations
# Retain ordered multiply/add rounding while vectorizing independent outputs.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# MPI is required for every invocation, including single-rank runs.
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

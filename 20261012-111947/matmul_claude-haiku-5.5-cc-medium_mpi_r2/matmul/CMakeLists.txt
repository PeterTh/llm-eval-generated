cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the benchmark always runs distributed across ranks
find_package(MPI REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
# -ffp-contract=off: no implicit fused multiply-add, so every element is rounded the same way
# regardless of tiling/vectorization and the result is identical for any rank count
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link MPI and math library
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

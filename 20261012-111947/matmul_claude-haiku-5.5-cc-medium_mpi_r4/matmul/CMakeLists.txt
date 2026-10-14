cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the benchmark is always built with distributed-memory parallelism
find_package(MPI REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations. -ffp-contract=off keeps every multiply and add separately rounded,
# so results are bit-identical across rank counts and to the serial k-ordered sum.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link MPI and math library
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

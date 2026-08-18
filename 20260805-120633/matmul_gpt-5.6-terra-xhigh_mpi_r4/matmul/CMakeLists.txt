cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is required: the executable always uses distributed-memory parallelism.
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

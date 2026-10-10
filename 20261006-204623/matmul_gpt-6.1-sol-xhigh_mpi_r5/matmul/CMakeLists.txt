cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "Use the MPI C interface from C++" FORCE)
find_package(MPI 3.0 REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Vectorize independent output elements without changing the summation order.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

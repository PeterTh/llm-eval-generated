cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Preserve the separate product rounding of GCC's vectorized serial dot product.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(matmul PRIVATE -ffp-contract=off)
endif()

target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

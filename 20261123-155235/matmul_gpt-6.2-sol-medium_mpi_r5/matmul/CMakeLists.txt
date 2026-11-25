cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Preserve GCC's separate multiply and add in the serial vectorized dot product.
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra
    $<$<CXX_COMPILER_ID:GNU>:-ffp-contract=off>)
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

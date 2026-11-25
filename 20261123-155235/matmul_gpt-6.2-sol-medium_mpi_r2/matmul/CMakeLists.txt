cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "Skip deprecated MPI C++ bindings")
find_package(MPI 3.0 REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Match the original dot product's rounding of products in GCC's vectorized loop.
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra
    $<$<CXX_COMPILER_ID:GNU>:-ffp-contract=off>)
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "Disable deprecated MPI C++ bindings")
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(spmv spmv.cpp)

# Keep multiply/add rounding consistent between paired rows and scalar tails.
target_compile_options(spmv PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

target_link_libraries(spmv PRIVATE MPI::MPI_CXX m)

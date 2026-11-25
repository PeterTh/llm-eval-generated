cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)
set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "Skip the deprecated MPI C++ bindings")
find_package(MPI REQUIRED COMPONENTS CXX)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

target_link_libraries(spmv PRIVATE MPI::MPI_CXX m)

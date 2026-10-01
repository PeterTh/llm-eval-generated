cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)
set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "Use the MPI C API from C++")
find_package(MPI REQUIRED COMPONENTS CXX)
target_compile_definitions(unstructured PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is required for every build, including single-rank runs.
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX m)

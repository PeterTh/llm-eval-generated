cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(stencil3d stencil3d.cpp)
target_compile_definitions(stencil3d PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is required, including for single-process runs.
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX m)

cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(stencil3d stencil3d.cpp)
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX)
target_compile_definitions(stencil3d PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# The stencil is bandwidth-bound; native ISA selection lets the compiler use
# the widest available vector loads while preserving IEEE arithmetic semantics.
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# Use the MPI C interface from C++ without the deprecated C++ bindings.
target_compile_definitions(stencil3d PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX m)

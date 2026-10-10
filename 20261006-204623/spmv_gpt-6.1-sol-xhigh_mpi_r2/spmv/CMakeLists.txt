cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(spmv spmv.cpp)

# Preserve separate multiply/add rounding when interleaving independent rows.
target_compile_options(spmv PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Use the MPI C interface without the deprecated C++ bindings.
target_compile_definitions(spmv PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(spmv PRIVATE MPI::MPI_CXX m)

cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Avoid deprecated MPI C++ bindings headers where present
target_compile_definitions(spmv PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)

# Link math library + MPI
target_link_libraries(spmv PRIVATE m MPI::MPI_CXX)

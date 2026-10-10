cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)
target_compile_definitions(nbody PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -fno-math-errno)

# MPI is required, including for single-rank executions.
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

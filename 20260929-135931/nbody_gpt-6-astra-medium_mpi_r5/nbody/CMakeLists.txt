cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# SIMD directives only: no OpenMP threads or runtime dependency.
# Keep strict floating-point sums; sqrt does not need to set errno.
# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -fno-math-errno -fopenmp-simd)

# MPI is required even for a single-rank run.
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

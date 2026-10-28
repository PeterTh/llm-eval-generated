cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations. -ffp-contract=off keeps floating-point results independent of
# FMA fusion choices, so output is bit-identical across rank counts and code paths.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# MPI for distributed-memory parallelism, plus math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

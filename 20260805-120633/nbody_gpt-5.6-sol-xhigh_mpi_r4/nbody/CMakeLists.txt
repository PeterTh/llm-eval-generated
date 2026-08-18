cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations. The force loop deliberately keeps ordered reductions so
# results remain bit-for-bit compatible with the original implementation.
target_compile_options(nbody PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    $<$<CXX_COMPILER_ID:GNU>:-fno-tree-loop-vectorize>
    $<$<CXX_COMPILER_ID:Clang>:-fno-vectorize>
)

# MPI is mandatory: this executable always uses the distributed implementation.
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

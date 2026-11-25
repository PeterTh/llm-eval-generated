cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX ON CACHE BOOL "Skip deprecated MPI C++ bindings")
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# The force kernel contracts selected operations explicitly.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -ffp-contract=off -Wall -Wextra)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

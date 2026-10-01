cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Allow vectorized sqrt without relaxing floating-point ordering or finiteness checks.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)

# MPI is required for every build, including single-rank runs.
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

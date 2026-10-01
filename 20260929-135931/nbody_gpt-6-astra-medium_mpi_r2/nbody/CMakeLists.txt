cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Allow vector sqrt without errno checks; retain strict floating-point sums.
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -fno-math-errno)

# MPI is required, including for single-process runs.
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

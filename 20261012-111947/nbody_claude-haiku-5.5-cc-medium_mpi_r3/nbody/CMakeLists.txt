cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the benchmark always runs distributed
find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
# -ffp-contract=off: fused multiply-adds are explicit in the code, so results don't depend on the optimizer
# -fno-math-errno: sqrt arguments are always positive here, and this lets the compiler vectorize sqrt
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -ffp-contract=off -fno-math-errno)

# Link MPI and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

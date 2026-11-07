cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations. Implicit FMA contraction is disabled: the force kernel
# spells out its fused multiply-adds explicitly so results are reproducible and
# bitwise identical to the original program regardless of compiler heuristics.
# -fno-math-errno lets sqrt() vectorize (its argument is always > 0 here).
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -fno-math-errno -Wall -Wextra)

# Link MPI and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

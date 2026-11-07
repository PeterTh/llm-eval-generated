cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations.
# -fno-math-errno lets sqrt vectorize (its argument is always > 0 here).
# -ffp-contract=off keeps FP rounding deterministic; the intended fused
# multiply-adds are written explicitly with std::fma in the source.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -ffp-contract=off -Wall -Wextra)

# Link MPI and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

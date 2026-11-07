cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the program is unconditionally parallelized with MPI.
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations
# -ffp-contract=off: FMA usage is made explicit in the source (std::fma) to
# reproduce the original program's rounding exactly.
# -fno-math-errno: lets sqrt() be inlined and vectorised (no errno side effect);
# it does not alter any floating-point result.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -fno-math-errno -Wall -Wextra)

# Link MPI and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

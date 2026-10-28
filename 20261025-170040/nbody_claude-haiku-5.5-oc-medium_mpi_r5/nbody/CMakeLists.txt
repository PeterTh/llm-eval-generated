cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the benchmark always runs with distributed-memory parallelism
find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations. Floating-point contraction is disabled so that every fused multiply-add
# in the kernels is explicit (std::fma) and results do not depend on the optimizer's choices.
# -fno-math-errno lets sqrt vectorize: distances are always >= SOFTENING, so no errno path is needed.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -fno-math-errno -Wall -Wextra)

# Link MPI and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

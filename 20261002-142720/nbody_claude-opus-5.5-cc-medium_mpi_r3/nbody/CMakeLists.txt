cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations
# Explicit std::fma calls in the source define rounding; no implicit contraction.
# sqrt arguments are always positive, so errno handling is dead and only blocks vectorization.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -fno-math-errno -Wall -Wextra)

# Link MPI and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

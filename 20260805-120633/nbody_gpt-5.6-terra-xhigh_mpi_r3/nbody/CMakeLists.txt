cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations.  The force denominator is strictly positive because of
# softening, so avoiding errno updates permits SIMD sqrt generation without
# changing any computed body state.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)

# MPI is required: bodies are distributed across ranks and positions are
# exchanged collectively once per timestep.
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

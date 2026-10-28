cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required unconditionally
find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
# -fno-math-errno lets sqrt be vectorized (results are unchanged)
# -ffp-contract=off keeps strict IEEE evaluation (no FMA fusion), so results are
# bitwise identical for any number of ranks and any kernel vectorization
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -ffp-contract=off -Wall -Wextra)

# Link MPI and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

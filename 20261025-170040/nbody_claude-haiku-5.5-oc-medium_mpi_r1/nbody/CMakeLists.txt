cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the benchmark always runs distributed across MPI ranks
find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
# -fno-math-errno lets sqrt be vectorized in the force kernel (results unchanged)
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -fno-math-errno)

# Link MPI and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

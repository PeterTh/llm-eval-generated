cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
# -fno-math-errno lets sqrt vectorize; results are unchanged (no errno side effects are used)
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)

# MPI for distributed-memory parallelism, math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX ON CACHE BOOL "Use MPI's C interface from C++")
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Vectorize independent bodies without reassociating force sums or using
# approximate reciprocals. The benchmark does not use math-library errno.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

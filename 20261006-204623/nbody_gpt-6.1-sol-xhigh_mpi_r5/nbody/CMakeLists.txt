cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Vectorize independent bodies without reassociating their force sums.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

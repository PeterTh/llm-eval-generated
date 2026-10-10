cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable SIMD square roots without relaxing floating-point summation order.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)

# Link MPI and the math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

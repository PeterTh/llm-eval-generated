cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the program always runs distributed
find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations. FP contraction is disabled so that results do not depend on how the
# compiler fuses multiply-adds in each loop shape; this keeps every rank count bit-identical.
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -fno-math-errno -ffp-contract=off)

# Link MPI and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

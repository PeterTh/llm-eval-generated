cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is required: the executable always initializes MPI and uses distributed
# Cartesian domain decomposition, including for a one-process run.
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX m)

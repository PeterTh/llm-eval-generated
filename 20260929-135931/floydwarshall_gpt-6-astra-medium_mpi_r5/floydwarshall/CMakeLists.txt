cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is required for every run, including a single rank.
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX m)

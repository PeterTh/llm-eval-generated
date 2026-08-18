cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is unconditional: every execution uses the distributed-memory path,
# including a normal single-rank launch.
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX m)

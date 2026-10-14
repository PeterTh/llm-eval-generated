cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# MPI is required for distributed memory parallelism
find_package(MPI REQUIRED)

# Link math library and MPI
target_link_libraries(black_scholes m MPI::MPI_CXX)

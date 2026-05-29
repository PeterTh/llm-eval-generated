cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI, math library
target_link_libraries(black_scholes MPI::MPI_CXX m)
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})

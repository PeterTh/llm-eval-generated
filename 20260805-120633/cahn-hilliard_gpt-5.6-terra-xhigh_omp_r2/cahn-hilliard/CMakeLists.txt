cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP is required: the benchmark's stencil workshares always execute in an
# OpenMP parallel region rather than falling back to a serial implementation.
find_package(OpenMP REQUIRED)
target_link_libraries(cahn_hilliard PRIVATE OpenMP::OpenMP_CXX m)

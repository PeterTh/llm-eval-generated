cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# This benchmark always uses OpenMP shared-memory parallelism.
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Link math and the required OpenMP runtime.
target_link_libraries(black_scholes PRIVATE m OpenMP::OpenMP_CXX)

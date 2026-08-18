cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# OpenMP is required: the benchmark always executes its independent option
# calculations through OpenMP work-sharing loops.
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Link math and OpenMP runtimes.
target_link_libraries(black_scholes PRIVATE m OpenMP::OpenMP_CXX)

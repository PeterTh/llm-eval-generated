cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

enable_language(CUDA)
add_executable(black_scholes black_scholes.cpp)
target_link_libraries(black_scholes PRIVATE cudart m)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -Wall -Wextra)

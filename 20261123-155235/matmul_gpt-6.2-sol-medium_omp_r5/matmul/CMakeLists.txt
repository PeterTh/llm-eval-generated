cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# GCC's original dot product rounds the vectorized products before adding.
target_compile_options(matmul PRIVATE $<$<CXX_COMPILER_ID:GNU>:-ffp-contract=off>)

target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX m)

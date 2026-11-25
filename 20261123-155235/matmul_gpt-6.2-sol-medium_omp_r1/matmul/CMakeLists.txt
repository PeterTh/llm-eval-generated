cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

find_package(OpenMP 4.0 REQUIRED COMPONENTS CXX)

# Use the same multiply and add rounding in the SIMD kernel and scalar reference.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX m)

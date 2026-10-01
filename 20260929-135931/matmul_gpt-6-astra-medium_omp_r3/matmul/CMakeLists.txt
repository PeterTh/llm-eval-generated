cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Optimize without contracting multiply/add operations or reassociating sums.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library
target_link_libraries(matmul PRIVATE m OpenMP::OpenMP_CXX)

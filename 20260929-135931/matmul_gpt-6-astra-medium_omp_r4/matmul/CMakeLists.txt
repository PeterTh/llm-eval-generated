cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# OpenMP is required for every build of this benchmark.
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Vectorize independent columns without changing the original summation order
# or introducing fused rounding in the multiply/add operations.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library
target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX m)

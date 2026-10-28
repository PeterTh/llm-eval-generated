cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Enable optimizations
# -ffp-contract=off keeps floating-point results independent of loop shape
# (no implicit FMA fusion), so output matches the reference serial loop exactly.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# OpenMP is required unconditionally
find_package(OpenMP REQUIRED)

# Link math library and OpenMP runtime
target_link_libraries(matmul m OpenMP::OpenMP_CXX)

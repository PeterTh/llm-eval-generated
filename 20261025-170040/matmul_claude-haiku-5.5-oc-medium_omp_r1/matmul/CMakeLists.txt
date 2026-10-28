cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Enable optimizations
# -ffp-contract=off keeps multiply and add as separate roundings, matching the
# original serial arithmetic so results are bitwise identical across versions.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# OpenMP is required
find_package(OpenMP REQUIRED)

# Link math library and OpenMP
target_link_libraries(matmul m OpenMP::OpenMP_CXX)

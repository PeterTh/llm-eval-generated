cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Enable optimizations
# -ffp-contract=off keeps multiply and add as separate roundings, so results do not
# depend on how the compiler happens to fuse them into FMA in each loop variant.
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra -ffp-contract=off)

# OpenMP is required for shared-memory parallelism
find_package(OpenMP REQUIRED)

# Link math library and OpenMP runtime
target_link_libraries(matmul m OpenMP::OpenMP_CXX)

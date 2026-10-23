cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations. FP contraction is disabled so that every element is
# accumulated with the same multiply-then-add rounding as the original code.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# OpenMP for multicore CPU shared-memory parallelism
target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX m)

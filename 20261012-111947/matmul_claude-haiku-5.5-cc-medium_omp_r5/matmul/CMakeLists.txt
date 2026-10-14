cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP is required: the benchmark is always built with shared-memory parallelism
find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
# -ffp-contract=off keeps each a*b + c as two rounded operations, so results do not
# depend on how the compiler happens to vectorize or fuse a given loop shape
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra -ffp-contract=off)

# Link math library and OpenMP runtime
target_link_libraries(matmul m OpenMP::OpenMP_CXX)

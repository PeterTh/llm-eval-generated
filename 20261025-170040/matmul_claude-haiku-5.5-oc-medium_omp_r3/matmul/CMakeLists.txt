cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP is required unconditionally
find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations. -ffp-contract=off keeps multiply-add rounding identical
# between the vectorized and scalar loop paths, so results are bitwise reproducible.
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra -ffp-contract=off)

# Link math library and OpenMP
target_link_libraries(matmul m OpenMP::OpenMP_CXX)

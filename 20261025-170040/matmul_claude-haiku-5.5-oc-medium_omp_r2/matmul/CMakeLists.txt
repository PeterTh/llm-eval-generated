cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations. Disable FMA contraction so that every multiply and add
# rounds exactly as written in the source; this keeps results bit-identical
# across matrix sizes and thread counts.
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra -ffp-contract=off)

# Link math library and OpenMP (unconditional)
target_link_libraries(matmul m OpenMP::OpenMP_CXX)

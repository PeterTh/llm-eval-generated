cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations. Floating point contraction is disabled so that the
# blocked parallel kernel reproduces the multiply/add sequence (and therefore
# the results) of the original scalar implementation.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library
target_link_libraries(matmul m OpenMP::OpenMP_CXX)

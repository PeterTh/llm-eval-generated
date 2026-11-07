cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations. FP contraction is disabled so the blocked kernel
# rounds exactly like the reference naive loop (explicit std::fma is still used
# where the reference does a fused multiply-add).
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library
target_link_libraries(matmul m OpenMP::OpenMP_CXX)

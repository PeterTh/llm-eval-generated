cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP 4.0 REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra
    # Keep SIMD microkernel accumulators in registers instead of spilling them.
    $<$<CXX_COMPILER_ID:GNU>:-fno-loop-unroll-and-jam>)

# Link the OpenMP runtime and math library
target_link_libraries(cholesky PRIVATE OpenMP::OpenMP_CXX m)

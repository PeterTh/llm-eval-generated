cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

# Keep the native tuning used by the original benchmark.  OpenMP SIMD only
# enables compiler vectorization directives; the parallel execution model is
# unconditionally MPI and no OpenMP runtime library is linked.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(matmul PRIVATE
        -O3
        -march=native
        -fopenmp-simd
        -Wall
        -Wextra
    )
endif()

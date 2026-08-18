cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX m)

# Keep optimization enabled independently of the selected CMake build type: the
# target is a benchmark, and its local BLAS-like kernels rely on vectorization.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cholesky PRIVATE
        -O3
        -march=native
        -ffp-contract=fast
        -Wall
        -Wextra
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cholesky PRIVATE -fopenmp-simd)
endif()

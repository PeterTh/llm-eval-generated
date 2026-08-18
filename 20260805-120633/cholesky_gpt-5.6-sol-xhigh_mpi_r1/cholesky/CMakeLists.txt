cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)

# The SIMD-only OpenMP flag enables portable vector directives without adding
# an OpenMP runtime dependency or changing the MPI-only execution model.
target_compile_options(cholesky PRIVATE
    -O3
    -march=native
    -fno-math-errno
    -fopenmp-simd
    -Wall
    -Wextra
)

target_link_libraries(cholesky PRIVATE MPI::MPI_CXX m)

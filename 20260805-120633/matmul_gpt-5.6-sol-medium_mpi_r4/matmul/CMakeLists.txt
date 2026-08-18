cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

# Optimize the per-rank cache-blocked multiplication kernel for the host CPU.
target_compile_options(matmul PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-march=native;-Wall;-Wextra>
)

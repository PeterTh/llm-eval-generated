cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Enable optimizations while retaining the original multiply-then-add rounding
# behavior used by the benchmark's serial kernel.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link MPI unconditionally: this executable is an MPI distributed-memory
# implementation, including when it runs with a single rank.
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

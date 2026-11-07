cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Enable optimizations (-ffp-contract=off keeps results bit-identical to the
# original mul-then-add loop; GCC would otherwise fuse into FMAs)
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link MPI and math library
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

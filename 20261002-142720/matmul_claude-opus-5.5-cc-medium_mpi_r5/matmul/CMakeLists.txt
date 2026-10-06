cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Enable optimizations. Floating-point contraction is disabled so that the
# kernel's explicit multiply/add sequence is preserved (bitwise-reproducible
# results identical to the sequential reference).
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link MPI and math library
target_link_libraries(matmul MPI::MPI_CXX m)

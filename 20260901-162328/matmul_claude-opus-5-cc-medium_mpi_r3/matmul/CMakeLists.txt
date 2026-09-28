cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Enable optimizations
# -ffp-contract=off keeps the multiply/add sequence unfused, so the blocked kernel
# reproduces the original scalar kernel's results bit-for-bit.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -funroll-loops -Wall -Wextra)

# Link math library and MPI
target_link_libraries(matmul m MPI::MPI_CXX)

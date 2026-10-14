cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the benchmark always runs distributed
find_package(MPI REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations. -ffp-contract=off keeps each multiply and add rounded
# separately, so results do not depend on FMA choices or on how rows are split across ranks
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra -ffp-contract=off)

# Link MPI and math library
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

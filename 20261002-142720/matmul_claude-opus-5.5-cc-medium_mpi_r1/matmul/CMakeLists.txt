cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Enable optimizations (no FMA contraction: every product is rounded and then
# added in ascending k order, exactly as written in the reference loop)
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link MPI and math library
target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

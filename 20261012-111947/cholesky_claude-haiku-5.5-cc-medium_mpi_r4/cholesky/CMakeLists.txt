cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the benchmark always runs distributed
find_package(MPI REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
# -ffp-contract=off keeps every multiply-add as a separate rounded operation, so the
# distributed result is bit-identical to the sequential arithmetic regardless of rank count
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra -ffp-contract=off)

# Link MPI and math library
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX m)

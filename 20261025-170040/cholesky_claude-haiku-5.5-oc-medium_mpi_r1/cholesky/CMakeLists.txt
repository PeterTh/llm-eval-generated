cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the benchmark always runs distributed across ranks
find_package(MPI REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations. Floating-point contraction is disabled so that multiply-add
# fusion is only where the code requests it explicitly (std::fma), keeping results deterministic.
target_compile_options(cholesky PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link MPI and math library
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX m)

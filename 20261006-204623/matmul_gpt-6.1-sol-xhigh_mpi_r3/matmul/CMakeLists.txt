cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Vectorize columns without changing the order of each dot product.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)
target_compile_definitions(matmul PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX m)

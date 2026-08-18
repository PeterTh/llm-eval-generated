cmake_minimum_required(VERSION 3.20)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# MPI partitions matrix rows among ranks, OpenMP prepares each rank's host data,
# and the CUDA Toolkit's tuned DGEMM implementation executes every local block.
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(matmul PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    m
)

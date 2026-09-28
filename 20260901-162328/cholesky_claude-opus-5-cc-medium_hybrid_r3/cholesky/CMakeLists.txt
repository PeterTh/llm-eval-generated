cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Hybrid parallelization: MPI + OpenMP + CUDA (cuBLAS)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

target_link_libraries(cholesky
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    m
)

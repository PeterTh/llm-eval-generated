cmake_minimum_required(VERSION 3.17)
project(cholesky_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Hybrid parallelization: MPI (multi-node/multi-GPU distribution) + OpenMP
# (host-side threading) + CUDA (cuBLAS/cuSOLVER GPU acceleration).
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link math, MPI, OpenMP and CUDA (cuBLAS/cuSOLVER) libraries
target_link_libraries(cholesky
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

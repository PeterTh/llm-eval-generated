cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA toolkit (for cudart and cublas headers/libs)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI, OpenMP, CUDA runtime, cuBLAS, and math library
target_link_libraries(matmul
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    m
)

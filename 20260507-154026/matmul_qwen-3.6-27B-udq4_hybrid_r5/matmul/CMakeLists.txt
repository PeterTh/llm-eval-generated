cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090 (Ampere)

add_executable(matmul matmul.cpp)

# Compile with nvcc for CUDA kernel support
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(matmul PRIVATE
    -O3
    -Xcompiler=-march=native
    -Xcompiler=-fopenmp
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)

target_link_libraries(matmul
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

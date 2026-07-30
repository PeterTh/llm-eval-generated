cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures for RTX 3090 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Use mpicxx as the CXX compiler for MPI support
set(CMAKE_CXX_COMPILER mpicxx)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and OpenMP support through nvcc
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra -Xcompiler -fopenmp)

# Link libraries
target_link_libraries(spmv MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

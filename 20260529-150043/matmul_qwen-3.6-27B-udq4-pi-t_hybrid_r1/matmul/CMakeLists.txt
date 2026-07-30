cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language support
enable_language(CUDA)
set(CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090 = SM 8.6

# Find MPI (CXX for host compilation)
find_package(MPI REQUIRED)

add_executable(matmul matmul.cpp)

# Compile matmul.cpp as CUDA source so nvcc handles the __global__ kernel
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Include directories: parent for common/, MPI headers
target_include_directories(matmul PRIVATE .. ${MPI_CXX_INCLUDE_DIRS})

# Host compiler flags (passed through nvcc's -Xcompiler): OpenMP + optimizations
target_compile_options(matmul PRIVATE
    -O3
    -Xcompiler=-march=native,-Wall,-Wextra,-fopenmp
)

# CUDA compiler flags
set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link: math, OpenMP, MPI
target_link_libraries(matmul m gomp ${MPI_CXX_LIBRARIES})

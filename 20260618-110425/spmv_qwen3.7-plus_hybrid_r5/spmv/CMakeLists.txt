cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

add_executable(spmv spmv.cu)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Add OpenMP flags for CUDA compilation
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fopenmp>)

# Link MPI, OpenMP, CUDA, and math library
target_link_libraries(spmv MPI::MPI_CXX OpenMP::OpenMP_CXX cudart m)

# Include directories
target_include_directories(spmv PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

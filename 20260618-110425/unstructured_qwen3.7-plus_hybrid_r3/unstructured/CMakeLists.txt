cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Compile the .cpp file as CUDA (it contains __global__ kernels)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations for host code
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# CUDA-specific options
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=-fopenmp -Xcompiler=-O3 -Xcompiler=-march=native>)

# Include directories for MPI
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(unstructured
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m)

cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED COMPONENTS CXX)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)

# Treat .cpp as CUDA source since it contains __global__ kernels
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# CUDA flags
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    --use_fast_math
    -arch=sm_86
    -Xcompiler=-fopenmp
    -Xcompiler=-march=native
>)

# Include directories
target_include_directories(unstructured PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Link libraries
target_link_libraries(unstructured MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

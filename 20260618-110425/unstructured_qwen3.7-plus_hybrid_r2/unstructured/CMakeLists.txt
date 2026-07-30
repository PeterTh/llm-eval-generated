cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture for RTX 3090 (compute 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Force the .cpp file to be compiled with nvcc (CUDA compiler)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Enable optimizations
target_compile_options(unstructured PRIVATE
    -O3 --use_fast_math -Xcompiler -fopenmp,-march=native,-Wall
)

# Link libraries
target_link_libraries(unstructured
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(nbody nbody.cpp)

# Treat .cpp file as CUDA source
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations with OpenMP support for host code
target_compile_options(nbody PRIVATE 
    -O3 
    -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp
)

# Link libraries
target_link_libraries(nbody 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX 
    CUDA::cudart
    m
)

# Set CUDA properties
set_target_properties(nbody PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)

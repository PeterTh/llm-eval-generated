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

# Treat .cpp file as CUDA source
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE 
    -O3 -march=native -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-fopenmp>
)

# Link libraries
target_link_libraries(unstructured 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

# Set CUDA architecture (adjust for your GPU)
set_property(TARGET unstructured PROPERTY CUDA_ARCHITECTURES 86)

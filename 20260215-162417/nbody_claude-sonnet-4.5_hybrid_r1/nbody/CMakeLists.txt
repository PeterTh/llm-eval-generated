cmake_minimum_required(VERSION 3.18)

# Set CUDA compiler before project() call
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Rename source file to .cu to enable CUDA compilation
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Set CUDA architectures (adjust based on target GPU)
set_target_properties(nbody PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON)

# Include directories
target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Compile options
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# OpenMP flags
target_compile_options(nbody PRIVATE ${OpenMP_CXX_FLAGS})

# Link libraries
target_link_libraries(nbody 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m)

cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Create executable with both C++ and CUDA sources
add_executable(qtclustering qtclustering.cpp cuda_kernels.cu)

# Compiler options
target_compile_options(qtclustering PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -fopenmp>
)

# Include directories and link libraries
target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(qtclustering 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

# Set CUDA architectures (adjust based on target GPU)
set_target_properties(qtclustering PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

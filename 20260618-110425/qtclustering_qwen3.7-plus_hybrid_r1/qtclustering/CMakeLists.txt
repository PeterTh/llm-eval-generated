cmake_minimum_required(VERSION 3.10)

project(qtclustering CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Treat the .cpp file as CUDA source so nvcc compiles it (it contains __global__ kernels)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

# Set CUDA architecture (RTX 3090 is compute capability 8.6)
set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)

# Include directories for MPI
target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-Xcompiler;-O3;-Xcompiler;-march=native;-Xcompiler;-Wall;-Xcompiler;-Wextra;-Xcompiler;-fopenmp>
)

# Link libraries
target_link_libraries(qtclustering
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
    cudart
)

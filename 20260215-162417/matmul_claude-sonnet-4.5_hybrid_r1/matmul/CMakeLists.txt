cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Rename source to .cu for CUDA compilation
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Set CUDA properties
set_target_properties(matmul PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(matmul 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (RTX 3090 = sm_86)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Add CUDA compile options
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>)

# Link libraries
target_link_libraries(cholesky 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    m
)

# Include directories
target_include_directories(cholesky PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

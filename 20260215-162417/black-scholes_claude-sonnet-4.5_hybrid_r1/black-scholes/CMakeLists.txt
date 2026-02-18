cmake_minimum_required(VERSION 3.18)

# Set CUDA architectures before project command
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Rename source file to .cu for CUDA compilation
add_executable(black_scholes black_scholes.cpp)

# Set the source file as CUDA
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Include directories
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(black_scholes 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

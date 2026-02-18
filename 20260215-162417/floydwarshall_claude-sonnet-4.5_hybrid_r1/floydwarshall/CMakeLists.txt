cmake_minimum_required(VERSION 3.18)

# Set CUDA path if not in standard location
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# Set CUDA architectures before project()
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Create executable
add_executable(floydwarshall floydwarshall.cpp)

# Set CUDA compilation for the source file
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-march=native,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Include directories
target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(floydwarshall 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cudart
    m
)

# Set CUDA separable compilation
set_target_properties(floydwarshall PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

cmake_minimum_required(VERSION 3.18)

# Set CUDA compiler before project declaration
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Set source file as CUDA
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Set CUDA properties
set_target_properties(spmv PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(spmv 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

# Include directories
target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})

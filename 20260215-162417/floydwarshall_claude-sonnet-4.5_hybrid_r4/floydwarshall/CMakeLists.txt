cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Rename source to .cu for CUDA compilation
add_executable(floydwarshall floydwarshall.cpp)

# Set CUDA properties
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Set CUDA architectures
set_property(TARGET floydwarshall PROPERTY CUDA_ARCHITECTURES 70 75 80 86)

# Link libraries
target_link_libraries(floydwarshall 
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

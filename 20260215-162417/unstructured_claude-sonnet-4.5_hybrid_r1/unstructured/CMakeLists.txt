cmake_minimum_required(VERSION 3.18)

# Set CUDA architectures before the project() call
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures")

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

# Set source properties for CUDA compilation
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link libraries
target_link_libraries(unstructured 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

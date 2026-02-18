cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17) # CUDA doesn't always support C++20 fully, but 17 is usually safe

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

# Host compiler options
target_compile_options(unstructured PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -fopenmp>
)

# CUDA compiler options
target_compile_options(unstructured PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -fopenmp>
)

# Link libraries
target_link_libraries(unstructured 
    PRIVATE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX 
)

# Treat .cpp file as CUDA source
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations and link libraries
target_compile_options(unstructured PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra ${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=native -Xcompiler -Wall,-Wextra,-fopenmp>
)

target_link_libraries(unstructured 
    PRIVATE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX 
    CUDA::cudart
    m
)

# Use nvcc for linking if CUDA code is present
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

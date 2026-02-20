cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Create executable - note that .cpp will be compiled as CUDA
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-march=native,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Set CUDA architectures (support common GPUs)
set_target_properties(unstructured PROPERTIES 
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(unstructured 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

# Include MPI headers
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})

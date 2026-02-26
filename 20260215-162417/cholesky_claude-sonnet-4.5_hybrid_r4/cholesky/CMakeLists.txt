cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

add_executable(cholesky cholesky.cpp)

# Set source properties to enable CUDA compilation
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Include directories
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations
target_compile_options(cholesky PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp,-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link libraries
target_link_libraries(cholesky 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    m
)

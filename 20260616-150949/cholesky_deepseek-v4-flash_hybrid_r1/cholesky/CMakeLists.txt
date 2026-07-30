cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required parallelisation frameworks
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

# Compile as CUDA so that __global__ / __device__ kernels are accepted
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimisations and OpenMP
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options;-O3;-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options;-fopenmp>
)

# Link libraries
target_link_libraries(cholesky PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

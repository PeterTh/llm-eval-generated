cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Treat .cpp as CUDA since it contains __global__ kernels
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations - pass host flags via -Xcompiler for nvcc
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 --compiler-options=-fopenmp,--std=c++17,-march=native,-Wall,-Wextra
        -lineinfo
    >
)

# Link MPI, OpenMP, CUDA, and math library
target_link_libraries(floydwarshall
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

# Set CUDA architecture for compute capability 8.6
set_target_properties(floydwarshall PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Compile .cpp as CUDA so NVCC handles kernel syntax
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "70")

# NVCC compile options: pass OpenMP and optimization flags to host compiler
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -Xcompiler=-fopenmp
        -Xcompiler=-march=native
        -Wall
        -Wextra
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3 -march=native -fopenmp -Wall -Wextra
    >
)

target_link_libraries(matmul PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

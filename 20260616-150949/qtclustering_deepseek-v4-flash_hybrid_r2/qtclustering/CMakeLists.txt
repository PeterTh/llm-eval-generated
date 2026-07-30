cmake_minimum_required(VERSION 3.20)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Compile as CUDA so nvcc handles __global__ kernels
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(qtclustering PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

target_compile_options(qtclustering PRIVATE -O3)

# Forward host compiler flags through nvcc
target_compile_options(qtclustering PRIVATE
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-march=native
    -Xcompiler=-fopenmp
)

target_link_libraries(qtclustering PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

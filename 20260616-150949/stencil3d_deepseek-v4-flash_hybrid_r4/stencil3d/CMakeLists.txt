cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "CUDA architectures to target")

add_executable(stencil3d stencil3d.cpp)

# Compile the source file as CUDA so nvcc handles kernels
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Pass OpenMP flags through nvcc's host compiler
target_compile_options(stencil3d PRIVATE
    -O3
    -Wall
    -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

target_link_libraries(stencil3d
    PUBLIC MPI::MPI_CXX
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC m)

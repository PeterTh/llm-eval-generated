cmake_minimum_required(VERSION 3.24)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The single translation unit contains CUDA kernels and is compiled by nvcc.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET cholesky PROPERTY CUDA_ARCHITECTURES native)
endif()

# Enable optimizations (host flags are forwarded through nvcc)
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)

target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cublas CUDA::cusolver m)

cmake_minimum_required(VERSION 3.24)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default GPU targets: Volta through Hopper (override with -DCMAKE_CUDA_ARCHITECTURES=...)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)

# The source contains CUDA kernels, so it is compiled by nvcc
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations. FP contraction is disabled on both host and device so that
# CPU-side and GPU-side arithmetic round identically.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-ffp-contract=off;-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler;-march=native;-Xcompiler;-Wall;-Xcompiler;-Wextra;-Xcompiler;-ffp-contract=off;-Xcompiler;-fopenmp>
)

# MPI for distribution, OpenMP for host-side phases, CUDA runtime for the kernels
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(spmv spmv.cpp)

# The single source file contains CUDA kernels; compile it with nvcc
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host-compiler flags forwarded through nvcc)
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra ${OpenMP_CXX_FLAGS}>)

# MPI headers for the CUDA-compiled translation unit (avoid MPI::MPI_CXX
# compile options such as -pthread that nvcc does not accept)
target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link with the host toolchain
set_target_properties(spmv PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(spmv PRIVATE CUDA::cudart MPI::MPI_CXX OpenMP::OpenMP_CXX m)

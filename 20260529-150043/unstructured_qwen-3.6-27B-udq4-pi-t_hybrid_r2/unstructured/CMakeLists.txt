cmake_minimum_required(VERSION 3.10)
project(unstructured)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use nvcc for CUDA + MPI + OpenMP hybrid compilation
set(CMAKE_CXX_COMPILER /usr/local/cuda/bin/nvcc)

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})

add_executable(unstructured unstructured.cpp)
target_include_directories(unstructured PRIVATE ../common)
target_link_libraries(unstructured ${MPI_CXX_LIBRARIES})
target_compile_options(unstructured PRIVATE -arch=sm_86 -O3 --expt-extended-lambda -Xcompiler=-fopenmp)

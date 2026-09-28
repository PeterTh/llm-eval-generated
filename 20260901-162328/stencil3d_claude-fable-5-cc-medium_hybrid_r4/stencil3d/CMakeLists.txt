cmake_minimum_required(VERSION 3.18)

# Prefer the toolkit at /usr/local/cuda (matches the installed driver) if the
# user did not specify a CUDA compiler.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(stencil3d stencil3d.cpp)

# The single source file contains CUDA kernels: compile it with nvcc.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations; forward host-compiler flags (incl. OpenMP) through nvcc.
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp -Wall -Wextra>)

# MPI include dirs/libraries are added manually to avoid passing host-only
# compile flags (e.g. -pthread) to nvcc.
target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(stencil3d PRIVATE ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} m)

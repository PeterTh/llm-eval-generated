cmake_minimum_required(VERSION 3.18)

# nvcc needs a supported host compiler (gcc <= 12 for CUDA 12.0)
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/g++-12)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# The single source file contains CUDA kernels; compile it with nvcc
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# -fmad=false (device) and -ffp-contract=off (host) keep GPU and CPU distance
# computations bit-identical, preserving the sequential algorithm's results.
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false -Xcompiler=-O3,-march=native,-fopenmp,-ffp-contract=off,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp -ffp-contract=off -Wall -Wextra>)

# Link MPI, OpenMP runtime, and math library
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

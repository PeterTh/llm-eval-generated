cmake_minimum_required(VERSION 3.10)

# Ensure a CUDA compiler is found even when nvcc is not on PATH.
if (NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

# CMake may not be able to auto-detect an architecture on headless nodes.
# Build a real binary for sm_70 and embed PTX for forward compatibility.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70-real;70-virtual" CACHE STRING "CUDA archs" FORCE)
endif()

project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xcompiler=-march=native -Xcompiler=-fopenmp>
)

target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

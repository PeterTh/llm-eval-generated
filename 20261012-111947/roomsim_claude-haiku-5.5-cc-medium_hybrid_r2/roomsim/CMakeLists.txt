cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU targets: Volta, Turing, Ampere, Hopper
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# The program mixes host code with CUDA kernels in one translation unit,
# so the source is compiled by nvcc (its host passes use the C++ compiler).
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Optimizations and warnings for the host code, OpenMP for the host threads.
# --fmad=false disables fused multiply-add on the device so every operation
# rounds separately, matching the expression order of the sequential code.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-march=native;-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-Wpedantic;-Xcompiler=-fopenmp>)

# MPI for distributing triangles across ranks, CUDA runtime, math library
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX CUDA::cudart m)
target_link_options(roomsim PRIVATE -fopenmp)

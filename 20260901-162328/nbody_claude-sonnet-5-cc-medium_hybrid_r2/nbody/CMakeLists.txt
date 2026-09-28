cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile the single source file as CUDA so it can unconditionally mix
# MPI (inter-node/inter-GPU), OpenMP (host-side parallelism) and CUDA
# (device kernels) in one translation unit.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

set_target_properties(nbody PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_compile_options(nbody PRIVATE
    -O3 --fmad=false
    -Xcompiler=-O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-fopenmp
    -Xcompiler=-ffp-contract=off
)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

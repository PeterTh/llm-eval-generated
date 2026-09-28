cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# The simulation kernels live in nbody.cpp and are compiled by nvcc
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations (host code is compiled through nvcc's host compiler)
target_compile_options(nbody PRIVATE
    # -fmad=false: the device code spells out the FMA contractions explicitly, so that results are
    # bit-identical to the (contracted) host reference
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extra-device-vectorization -fmad=false
                               -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

cmake_minimum_required(VERSION 3.24)
# CUDA 12.0 supports GCC through version 12. Respect an explicit host compiler.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(ROOMSIM_CUDA_HOST_COMPILER NAMES g++-12 g++)
    set(CMAKE_CUDA_HOST_COMPILER "${ROOMSIM_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler")
endif()
# Optimize for the build node's GPUs by default; cluster/cross builds can set
# CMAKE_CUDA_ARCHITECTURES (or CUDAARCHS) for their target accelerator fleet.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Target CUDA architectures")
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source file and executable name.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-march=native,-Wall,-Wextra,-ffp-contract=off>)
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

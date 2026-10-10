cmake_minimum_required(VERSION 3.20)

# Native code for common cluster GPUs, plus PTX for newer devices.
# Cluster-specific builds can override CMAKE_CUDA_ARCHITECTURES.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86-real 70-virtual)
    set(STENCIL3D_DEFAULT_ARCHITECTURES ON)
endif()
project(stencil3d_benchmark LANGUAGES CXX CUDA)
if(STENCIL3D_DEFAULT_ARCHITECTURES AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 90-real)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the original source filename and executable name.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
set_target_properties(stencil3d PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_RUNTIME_LIBRARY Shared)
target_compile_definitions(stencil3d PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--prec-div=true;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

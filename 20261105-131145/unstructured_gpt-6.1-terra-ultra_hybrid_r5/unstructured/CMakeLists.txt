cmake_minimum_required(VERSION 3.18)
# Preserve an architecture choice supplied by the user or toolchain.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(unstructured_default_cuda_architectures ON)
endif()
project(unstructured_benchmark LANGUAGES CXX CUDA)

if(unstructured_default_cuda_architectures)
    # Include native code for common cluster GPUs and PTX for newer devices.
    # CUDA 13 no longer supports offline compilation for pre-Turing GPUs.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        set(unstructured_cuda_architectures 75-real 80-real 86-real 90-real 75-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        set(unstructured_cuda_architectures 60-real 70-real 80-real 86-real 90-real 60-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
        set(unstructured_cuda_architectures 60-real 70-real 80-real 86-real 60-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
        set(unstructured_cuda_architectures 60-real 70-real 80-real 60-virtual)
    else()
        set(unstructured_cuda_architectures 60-real 70-real 75-real 60-virtual)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${unstructured_cuda_architectures}" CACHE STRING
        "CUDA architectures to build for" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The CUDA kernels live in the existing source file.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler ${OpenMP_CXX_FLAGS}>)

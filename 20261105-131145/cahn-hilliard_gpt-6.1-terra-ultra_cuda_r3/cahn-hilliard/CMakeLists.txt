cmake_minimum_required(VERSION 3.18)

# Respect an architecture list supplied by the caller.  Otherwise build native
# code for common GPUs as well as PTX for forward compatibility.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(_cahn_default_cuda_architectures TRUE)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

if(_cahn_default_cuda_architectures)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 75-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        set(CMAKE_CUDA_ARCHITECTURES 60-real 70-real 75-real 80-real 86-real 89-real 90-real 60-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
        set(CMAKE_CUDA_ARCHITECTURES 60-real 70-real 75-real 80-real 86-real 60-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
        set(CMAKE_CUDA_ARCHITECTURES 60-real 70-real 75-real 80-real 60-virtual)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 60-real 70-real 75-real 60-virtual)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING
        "CUDA architectures for the benchmark" FORCE)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

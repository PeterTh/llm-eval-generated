cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(use_detected_cuda_architectures ON)
endif()

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

# Keep the compiler's portable default and add native code for visible GPUs.
if(use_detected_cuda_architectures AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
    set(CMAKE_CUDA_ARCHITECTURES
        "${CMAKE_CUDA_ARCHITECTURES};${CMAKE_CUDA_ARCHITECTURES_NATIVE}"
        CACHE STRING "CUDA architectures" FORCE)
endif()

# Compile CUDA in the existing source file.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_compile_options(cahn_hilliard PRIVATE -O3)

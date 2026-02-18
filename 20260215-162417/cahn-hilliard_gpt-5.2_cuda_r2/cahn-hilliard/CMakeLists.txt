cmake_minimum_required(VERSION 3.10)

# Ensure CMake can find NVCC even when it's not in PATH
if (NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()

# Target Ampere by default (RTX 3090 = sm_86) for maximum performance on this system
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")
endif()

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile this translation unit with NVCC (keep the existing filename/executable name)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
)

# Link math library
target_link_libraries(cahn_hilliard m)

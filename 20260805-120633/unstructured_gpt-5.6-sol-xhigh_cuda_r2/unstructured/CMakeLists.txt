cmake_minimum_required(VERSION 3.24)

# CUDA 12.0 does not accept GCC 13 as a host compiler. Prefer the newest
# commonly available compatible GNU compiler when the caller did not select
# one explicitly.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(CUDA_HOST_COMPILER NAMES g++-12 g++-11 g++)
    if(CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${CUDA_HOST_COMPILER}" CACHE FILEPATH
            "Host compiler used by NVCC")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
        "CUDA architectures to compile for")
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Optimize both device code and NVCC's host-side compilation.
target_compile_options(unstructured PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;--restrict;-Xcompiler=-O3,-march=native,-Wall,-Wextra>")
set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# Link math library
target_link_libraries(unstructured PRIVATE m)

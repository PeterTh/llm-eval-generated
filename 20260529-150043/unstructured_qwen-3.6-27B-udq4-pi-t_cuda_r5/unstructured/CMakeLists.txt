cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (detect from device, default to 86 for Ampere)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for host code
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# CUDA compilation flags: optimize for performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra -lineinfo")

# Link math library (host side)
target_link_libraries(unstructured m)

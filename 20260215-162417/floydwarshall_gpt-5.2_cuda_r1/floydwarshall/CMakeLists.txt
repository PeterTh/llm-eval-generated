cmake_minimum_required(VERSION 3.10)

# Ensure a usable CUDA compiler even if nvcc isn't on PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

project(floydwarshall_benchmark LANGUAGES CXX CUDA)

# NVCC support for C++20 can vary; this code doesn't require C++20 features.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build a tuned binary for the local GPU when available.
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(floydwarshall floydwarshall.cpp)

# Compile the existing .cpp with NVCC (no new files).
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)

# Link math library (CUDA runtime is linked automatically by CMake for CUDA targets).
target_link_libraries(floydwarshall PRIVATE m)

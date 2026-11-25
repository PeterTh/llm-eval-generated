cmake_minimum_required(VERSION 3.23)

# Prefer the system's selected CUDA toolkit when no compiler was requested.
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(CMAKE_CUDA_COMPILER nvcc HINTS /usr/local/cuda/bin REQUIRED)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
project(floydwarshall_benchmark LANGUAGES CUDA)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
set_target_properties(floydwarshall PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(floydwarshall PRIVATE -O3)

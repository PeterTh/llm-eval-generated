cmake_minimum_required(VERSION 3.23)

# Ship device code for supported GPU generations without requiring a GPU at build time.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(nbody_benchmark LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)

# Contract only the explicitly fused operations in the simulation.
target_compile_options(nbody PRIVATE -O3 --fmad=false -Xcompiler=-Wall,-Wextra)

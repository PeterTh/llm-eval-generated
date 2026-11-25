cmake_minimum_required(VERSION 3.23)

# Build device code for supported GPU generations without requiring a GPU at build time.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(cahn_hilliard_benchmark LANGUAGES CUDA)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(cahn_hilliard PRIVATE -O3 --fmad=false -Xcompiler=-Wall,-Wextra)

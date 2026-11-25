cmake_minimum_required(VERSION 3.20)
# Prefer native device code when CMake detects a GPU on the build host.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(prefer_native_arch ON)
endif()
project(nbody_benchmark LANGUAGES CUDA)
if(prefer_native_arch AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" CACHE STRING "CUDA architectures to build for" FORCE)
endif()

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
set_target_properties(nbody PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(nbody PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra)

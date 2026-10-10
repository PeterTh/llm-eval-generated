cmake_minimum_required(VERSION 3.24)

# CUDA 12.0 does not accept GCC 13. Respect an explicitly selected CUDA
# host compiler, otherwise prefer the compatible compiler when available.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(CUDA_HOST_GXX NAMES g++-12)
    if(CUDA_HOST_GXX)
        set(CMAKE_CUDA_HOST_COMPILER "${CUDA_HOST_GXX}")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename and executable name.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

# Preserve the sequence of double-precision operations used by the reference.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra>)

# Link math library
target_link_libraries(unstructured m)

cmake_minimum_required(VERSION 3.18)

# CUDA 12.0 does not support GCC 13 as a host compiler.  Prefer the installed
# supported compiler when configuring on systems that provide it, while still
# allowing callers to explicitly select a different CUDA host compiler.
if (NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++-12")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12" CACHE FILEPATH
        "CUDA host compiler")
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# The numerical update uses double precision, so do not enable CUDA fast math.
# Disabling contraction preserves the original sequence of floating-point
# operations while the element updates themselves run in parallel on the GPU.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

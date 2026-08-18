cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark is run on Ampere GPUs.  Building a native SASS image avoids
# JIT compilation and lets nvcc use the full double-precision instruction set.
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(unstructured unstructured.cpp)

# Compile the existing source file with nvcc so the executable always uses the
# CUDA implementation; no CPU execution fallback is built.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations.  fmad is disabled to retain the operation ordering of
# the original scalar double-precision implementation exactly.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>
)

# Link math library
target_link_libraries(unstructured m)

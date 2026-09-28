cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Compile nbody.cpp as CUDA source so the simulation runs unconditionally on the GPU
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations
# -fmad=false avoids fused multiply-add contraction so results match the
# original scalar (non-FMA) floating-point semantics as closely as possible.
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false>)

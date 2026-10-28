cmake_minimum_required(VERSION 3.24)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures (Volta through Hopper) unless overridden.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

add_executable(qtclustering qtclustering.cpp)

# The source is CUDA (host code plus __global__ kernels).
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Host code: keep original flags; disable FP contraction so host and device
# compute identical (unfused) distances.
# Device code: disable FMA contraction for the same reason.
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-march=native,-Wall,-Wextra,-ffp-contract=off>
)

# Link math library
target_link_libraries(qtclustering m)

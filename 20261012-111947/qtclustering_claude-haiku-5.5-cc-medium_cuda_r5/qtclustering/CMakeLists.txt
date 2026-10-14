cmake_minimum_required(VERSION 3.24)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU present on this machine unless the user chooses otherwise
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The program is CUDA code, so compile the existing source file as CUDA
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

# Host compiler flags are forwarded through nvcc. -ffp-contract=off keeps host
# floating-point arithmetic free of fused multiply-adds so that it matches the
# device code bit for bit.
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --compiler-options=-O3,-march=native,-Wall,-Wextra,-ffp-contract=off>)

# Link math library
target_link_libraries(qtclustering m)

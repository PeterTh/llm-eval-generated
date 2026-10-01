cmake_minimum_required(VERSION 3.18)
# Supply portable GPU code plus PTX unless the caller selects architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80 86)
endif()
project(cholesky_benchmark LANGUAGES CXX CUDA)

# Compile the existing source as CUDA without changing the executable name.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)
set_target_properties(cholesky PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

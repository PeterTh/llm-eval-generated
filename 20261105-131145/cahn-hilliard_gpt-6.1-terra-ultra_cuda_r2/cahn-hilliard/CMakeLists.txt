cmake_minimum_required(VERSION 3.18)
# Produce native code for the installed toolkit's supported GPUs when CMake
# supports it, while allowing callers to select their own architectures.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23 AND
   NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES all)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

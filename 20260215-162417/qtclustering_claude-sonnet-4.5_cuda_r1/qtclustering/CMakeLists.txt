cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Rename source file to .cu for CUDA compilation
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

# Set CUDA architectures - use native or common architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
endif()

set_target_properties(qtclustering PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(qtclustering PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link math library
target_link_libraries(qtclustering m cudart)

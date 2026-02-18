cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Rename cpp to cu for CUDA compilation
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Set CUDA architectures - using common architectures
set_target_properties(cahn_hilliard PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86")

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link math library
target_link_libraries(cahn_hilliard m cudart)

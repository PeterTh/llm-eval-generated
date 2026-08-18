cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Compile device code for the build machine unless the caller selected an
# architecture explicitly (for example, -DCMAKE_CUDA_ARCHITECTURES=80;90).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET black_scholes PROPERTY CUDA_ARCHITECTURES native)
endif()

target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;--restrict>)
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

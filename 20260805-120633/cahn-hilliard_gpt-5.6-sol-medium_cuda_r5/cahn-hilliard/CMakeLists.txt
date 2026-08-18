cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing filename while compiling it as CUDA source.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)

# Native selects the installed GPU when supported by CMake; the fallback keeps
# this project usable with older CMake versions paired with CUDA 12.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set_property(TARGET cahn_hilliard PROPERTY CUDA_ARCHITECTURES native)
else()
    set_property(TARGET cahn_hilliard PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--extra-device-vectorization;-Xcompiler=-Wall,-Wextra>
)

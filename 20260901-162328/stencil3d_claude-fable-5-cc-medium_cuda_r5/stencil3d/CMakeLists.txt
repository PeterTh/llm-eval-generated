cmake_minimum_required(VERSION 3.18)

# nvcc rejects newer host compilers; prefer g++-12 when the default gcc is too new
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++-12")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12")
endif()

project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(stencil3d stencil3d.cpp)

# The source is CUDA code kept in a .cpp file
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link math library
target_link_libraries(stencil3d m)

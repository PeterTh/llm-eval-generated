cmake_minimum_required(VERSION 3.18)

# Respect explicit toolchains, otherwise prefer the active toolkit installation.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL "")
    find_program(MATMUL_NVCC nvcc HINTS ENV CUDA_PATH /usr/local/cuda PATH_SUFFIXES bin)
    if(MATMUL_NVCC)
        set(CMAKE_CUDA_COMPILER "${MATMUL_NVCC}" CACHE FILEPATH "CUDA compiler")
    endif()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL ""
   AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename while compiling its kernels with nvcc.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)

# Link math library
target_link_libraries(matmul m)

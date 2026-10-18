cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

# Keep separate double-precision multiply/add operations, as in the CPU loop.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(matmul m)

cmake_minimum_required(VERSION 3.24)

# Tune for the build machine's GPUs unless an architecture was explicitly supplied.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename and executable name.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

# Retain double precision; do not enable CUDA's fast-math transformations.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)

# Link math library
target_link_libraries(matmul m)

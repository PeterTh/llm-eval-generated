cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Compile the existing source as CUDA.  The explicit SM target lets NVCC tune
# the tiled kernel for the Ampere GPUs used by this benchmark environment.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>)

# CUDA is deliberately required: this benchmark unconditionally executes the
# GPU-parallel implementation.
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries(matmul m)

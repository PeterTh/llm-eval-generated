cmake_minimum_required(VERSION 3.24)

# RTX 3090-class Ampere GPUs use compute capability 8.6.  Callers targeting a
# different GPU can still override CMAKE_CUDA_ARCHITECTURES at configure time.
# Supplying a real architecture avoids runtime JIT and enables Ampere codegen.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable aggressive optimization for both host setup code and the CUDA kernel.
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(spmv PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

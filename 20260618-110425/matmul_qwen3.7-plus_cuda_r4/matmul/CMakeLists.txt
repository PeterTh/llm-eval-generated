cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Treat .cpp file as CUDA source (it contains CUDA kernels)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(matmul PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-use_fast_math;-Xcompiler;-O3;-Xcompiler;-march=native;-Xcompiler;-Wall>
)

# Link CUDA runtime and math
target_link_libraries(matmul cudart m)

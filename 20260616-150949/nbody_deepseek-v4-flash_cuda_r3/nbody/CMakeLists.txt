cmake_minimum_required(VERSION 3.11)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target compute capability for RTX 3090 (Ampere SM 8.6)
set(CMAKE_CUDA_ARCHITECTURES "86")

add_executable(nbody nbody.cpp)

# Force nbody.cpp to be compiled as CUDA so that kernel code is handled by nvcc
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Host compiler optimizations
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# CUDA device compiler optimizations
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Wall -Wextra>
)

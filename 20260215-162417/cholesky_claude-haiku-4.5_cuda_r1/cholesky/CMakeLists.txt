cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (RTX 3090 = sm_86)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Compile as CUDA
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>)

# Link math library
target_link_libraries(cholesky m)

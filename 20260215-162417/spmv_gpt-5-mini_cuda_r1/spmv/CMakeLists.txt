cmake_minimum_required(VERSION 3.10)

# Set target CUDA architectures to avoid autodetect failures in CI environments
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Compile spmv.cpp as CUDA source so nvcc is used for device kernels
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
# Add CUDA-specific compiler flags
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(spmv m)

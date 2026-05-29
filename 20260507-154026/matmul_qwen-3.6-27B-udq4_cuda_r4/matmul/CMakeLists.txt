cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Treat .cpp as CUDA source so nvcc compiles it (contains __global__ kernels)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations
set_target_properties(matmul PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86;89")
target_compile_options(matmul PRIVATE -O3 -Xcompiler=-march=native -Wall -Wextra)

# Link math library
target_link_libraries(matmul m)

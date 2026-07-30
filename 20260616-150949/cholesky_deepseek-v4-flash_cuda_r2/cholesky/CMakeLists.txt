cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# Compile as CUDA source so that __global__ / __device__ keywords are accepted
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Target-specific CUDA architecture (RTX 3090 = Ampere, sm_86)
set_target_properties(cholesky PROPERTIES CUDA_ARCHITECTURES "86")

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(cholesky m)

cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

cuda_add_executable(matmul matmul.cpp)

# Enable optimizations
# CUDA flags are set via CUDA_NVCC_FLAGS
set(CUDA_NVCC_FLAGS "-O3 -arch=sm_60")

# Link math library
target_link_libraries(matmul m)

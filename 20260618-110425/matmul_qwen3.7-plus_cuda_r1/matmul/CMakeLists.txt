cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cu)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3)
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")

# Link math library
target_link_libraries(matmul m)

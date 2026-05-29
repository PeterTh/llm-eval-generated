cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

add_executable(spmv spmv.cu)

# Enable optimizations
set_target_properties(spmv PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
target_compile_options(spmv PRIVATE -O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra)

# Link math library
target_link_libraries(spmv m)

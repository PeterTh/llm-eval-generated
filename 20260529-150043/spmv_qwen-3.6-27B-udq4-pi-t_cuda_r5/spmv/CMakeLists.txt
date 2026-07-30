cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture for performance (compute capability 6.0+)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

add_executable(spmv spmv.cu)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -Wall -Wextra)

# Link math library
target_link_libraries(spmv m)

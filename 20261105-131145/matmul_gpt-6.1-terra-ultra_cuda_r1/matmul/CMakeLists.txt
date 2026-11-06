cmake_minimum_required(VERSION 3.17)
project(matmul_benchmark LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# cuBLAS and the CUDA runtime are supplied by the CUDA toolkit.
target_link_libraries(matmul PRIVATE CUDA::cublas CUDA::cudart)

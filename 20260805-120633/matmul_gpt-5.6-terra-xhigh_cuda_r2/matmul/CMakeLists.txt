cmake_minimum_required(VERSION 3.17)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# CUDA's cuBLAS supplies the GPU DGEMM implementation.  It is a required
# dependency: there is intentionally no CPU multiplication fallback.
target_link_libraries(matmul PRIVATE CUDA::cudart CUDA::cublas m)

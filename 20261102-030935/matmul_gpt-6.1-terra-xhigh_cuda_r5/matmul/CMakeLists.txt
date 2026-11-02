cmake_minimum_required(VERSION 3.17)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

find_package(CUDAToolkit REQUIRED)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(matmul PRIVATE CUDA::cublas CUDA::cudart)

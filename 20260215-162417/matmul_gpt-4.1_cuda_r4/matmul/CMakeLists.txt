cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(matmul_benchmark LANGUAGES CXX CUDA)

add_executable(matmul matmul.cpp)
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)

# Link math library
target_link_libraries(matmul m cuda)

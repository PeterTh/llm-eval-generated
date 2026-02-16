cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the single source file with NVCC by marking it as CUDA language
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations for host and device
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Ensure separable compilation for better performance on some toolchains
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link math library
target_link_libraries(matmul m)

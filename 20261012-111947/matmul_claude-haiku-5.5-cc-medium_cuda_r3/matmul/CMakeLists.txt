cmake_minimum_required(VERSION 3.24)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPUs present on this machine
set(CMAKE_CUDA_ARCHITECTURES native)

# The benchmark is GPU-parallel: compile matmul.cpp with nvcc
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations (host flags are forwarded through nvcc's -Xcompiler)
target_compile_options(matmul PRIVATE -O3 -Xcompiler -march=native -Xcompiler -Wall -Xcompiler -Wextra)

# Link math library
target_link_libraries(matmul m)

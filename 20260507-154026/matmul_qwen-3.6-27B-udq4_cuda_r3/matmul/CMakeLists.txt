cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture for maximum performance (compute 80 = Ampere, adjust as needed)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)

# Link math library
target_link_libraries(matmul m)

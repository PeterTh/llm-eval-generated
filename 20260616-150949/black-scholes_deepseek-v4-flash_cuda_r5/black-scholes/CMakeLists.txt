cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# Compile the source file with nvcc for CUDA support
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set_target_properties(black_scholes PROPERTIES CUDA_ARCHITECTURES "86")

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(black_scholes m)

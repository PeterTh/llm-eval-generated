cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 50)

add_executable(black_scholes black_scholes.cpp)

# Force nvcc to compile the .cpp source (it contains CUDA device code)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Compile options for nvcc
target_compile_options(black_scholes PRIVATE -O3 -Xcompiler=-Wall,-Wextra)

# Link CUDA runtime
target_link_libraries(black_scholes PRIVATE cudart)

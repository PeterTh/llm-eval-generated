cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture for RTX 3090
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(black_scholes black_scholes.cpp)

# Treat the .cpp file as CUDA source since it contains CUDA code
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# CUDA specific flags
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>)

# Link math library and CUDA runtime
target_link_libraries(black_scholes m ${CUDA_LIBRARIES})

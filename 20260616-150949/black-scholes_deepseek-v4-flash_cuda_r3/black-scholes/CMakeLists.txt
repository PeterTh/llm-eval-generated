cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architecture (RTX 3090 is compute 8.6, add more for portability)
set(CMAKE_CUDA_ARCHITECTURES "86")

add_executable(black_scholes black_scholes.cpp)

# Compile as CUDA to enable kernel compilation
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable host optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(black_scholes PRIVATE m)

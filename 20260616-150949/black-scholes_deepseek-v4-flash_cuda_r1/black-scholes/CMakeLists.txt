cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# Compile source file with CUDA compiler
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Set CUDA architecture for RTX 3090 (sm_86) and other common GPUs
set_target_properties(black_scholes PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;89;90"
)

# Link math library
target_link_libraries(black_scholes m)

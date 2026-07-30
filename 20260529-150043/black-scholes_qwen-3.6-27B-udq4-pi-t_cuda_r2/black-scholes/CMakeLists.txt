cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build the executable - compile .cpp with NVCC for CUDA support
add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# CUDA architecture targets
set_target_properties(black_scholes PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86"
)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -Wall -Wextra)

# Link math library
target_link_libraries(black_scholes PRIVATE m)

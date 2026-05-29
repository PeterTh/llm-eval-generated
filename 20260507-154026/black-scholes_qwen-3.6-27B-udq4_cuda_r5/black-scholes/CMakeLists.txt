cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set default architecture if not specified
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif()

add_executable(black_scholes black_scholes.cpp)

# Force nvcc for the .cpp file since it contains CUDA code
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for both host and device
set_target_properties(black_scholes PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(black_scholes PRIVATE
    -O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -lineinfo
)

# Link math library
target_link_libraries(black_scholes m)

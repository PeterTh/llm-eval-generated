cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures for common GPUs
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")

add_executable(black_scholes black_scholes.cu)

# Enable optimizations for CUDA
set_target_properties(black_scholes PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link math library
target_link_libraries(black_scholes m)

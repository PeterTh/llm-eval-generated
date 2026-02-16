cmake_minimum_required(VERSION 3.10)
# Set a default CUDA architecture to avoid detection failures in some environments
set(CMAKE_CUDA_ARCHITECTURES 70)
project(black_scholes_benchmark LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# Treat the .cpp as CUDA source so nvcc compiles device code in it
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for host and device
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xfatbin -lineinfo>
)

# Link math and CUDA runtime libraries
target_link_libraries(black_scholes m cudart)

# Ensure position-independent code off for maximum perf on some platforms
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

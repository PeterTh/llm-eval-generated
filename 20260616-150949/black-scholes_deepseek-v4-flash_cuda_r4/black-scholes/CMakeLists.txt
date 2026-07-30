cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations for host code
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# CUDA compiler flags - target sm_86 (Ampere) for GPU with compute capability 8.6
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_86 --expt-relaxed-constexpr>)

# Ensure CUDA is enabled for the source
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

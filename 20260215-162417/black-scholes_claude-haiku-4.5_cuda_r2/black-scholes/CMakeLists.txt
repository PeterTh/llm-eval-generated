cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 60-real 61-real 70-real 75-real 80-real 86-real 89-real)

add_executable(black_scholes black_scholes.cu)

# Enable optimizations for C++
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)

# Enable optimizations for CUDA
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-use_fast_math;--ptxas-options=-v>)

# Link math library
target_link_libraries(black_scholes PRIVATE m)
